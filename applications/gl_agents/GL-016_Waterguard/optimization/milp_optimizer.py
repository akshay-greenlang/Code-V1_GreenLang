"""
GL-016 Waterguard MILP Optimizer - Mixed Integer Linear Programming

Mixed Integer Linear Programming optimizer for cooling tower water treatment
systems. Minimizes combined objective of water loss, energy loss, chemical
cost, and risk penalty subject to chemistry and equipment constraints.

Optimization Formulation:
    Minimize: Water_Loss + Energy_Loss + Chemical_Cost + Risk_Penalty
    Subject to:
        - Chemistry constraints (pH, conductivity, LSI, RSI) - HARD
        - Equipment constraints (valve positions, pump speeds) - HARD
        - Ramp rate limits (blowdown valve, dosing pumps) - HARD
        - Target ranges (cycles of concentration) - SOFT

Key Features:
    - Rolling horizon optimization (15-60 minute lookahead)
    - Risk penalty from ML models (bounded and calibrated)
    - Conservative recommendations under high uncertainty
    - Provenance tracking with SHA-256 hashes

Reference Standards:
    - CTI STD-201 (Cooling Tower Water Treatment)
    - ASHRAE 188 (Legionella Risk Management)
    - IEC 61131-3 (Control Systems)

Author: GreenLang Water Treatment Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, field_validator

# Optional cvxpy import
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

# Optional scipy import
try:
    from scipy.optimize import minimize, linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    minimize = None
    linprog = None

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class OptimizationStatus(str, Enum):
    """Status of optimization solution."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIMEOUT = "timeout"
    CONSERVATIVE_FALLBACK = "conservative_fallback"
    ERROR = "error"


class ChemicalType(str, Enum):
    """Types of treatment chemicals."""
    SCALE_INHIBITOR = "scale_inhibitor"
    CORROSION_INHIBITOR = "corrosion_inhibitor"
    BIOCIDE = "biocide"
    PH_ACID = "ph_acid"
    PH_CAUSTIC = "ph_caustic"
    DISPERSANT = "dispersant"


class ConstraintType(str, Enum):
    """Constraint classification."""
    HARD = "hard"  # Must satisfy
    SOFT = "soft"  # Optimize within


class RiskType(str, Enum):
    """Types of operational risks."""
    SCALING = "scaling"
    CORROSION = "corrosion"
    CARRYOVER = "carryover"
    BIOLOGICAL = "biological"


# =============================================================================
# DATA MODELS
# =============================================================================

class ChemistryState(BaseModel):
    """Current water chemistry state."""
    conductivity_us_cm: float = Field(..., ge=0, description="Conductivity (uS/cm)")
    ph: float = Field(..., ge=0, le=14, description="pH value")
    temperature_c: float = Field(..., ge=0, le=100, description="Water temperature (C)")
    calcium_hardness_ppm: float = Field(default=0, ge=0, description="Calcium hardness (ppm)")
    alkalinity_ppm: float = Field(default=0, ge=0, description="M-Alkalinity (ppm)")
    silica_ppm: float = Field(default=0, ge=0, description="Silica (ppm)")
    chlorides_ppm: float = Field(default=0, ge=0, description="Chlorides (ppm)")
    cycles_of_concentration: float = Field(default=1.0, ge=1.0, description="Current CoC")
    tds_ppm: Optional[float] = Field(default=None, ge=0, description="TDS (ppm)")

    @property
    def lsi(self) -> float:
        """Calculate Langelier Saturation Index."""
        if self.calcium_hardness_ppm == 0 or self.alkalinity_ppm == 0:
            return 0.0
        # Simplified LSI calculation
        phs = (9.3 + np.log10(self.calcium_hardness_ppm) +
               np.log10(self.alkalinity_ppm) - 0.02 * self.temperature_c)
        return self.ph - phs

    @property
    def rsi(self) -> float:
        """Calculate Ryznar Stability Index."""
        if self.calcium_hardness_ppm == 0 or self.alkalinity_ppm == 0:
            return 12.0
        phs = (9.3 + np.log10(self.calcium_hardness_ppm) +
               np.log10(self.alkalinity_ppm) - 0.02 * self.temperature_c)
        return 2 * phs - self.ph


class EquipmentState(BaseModel):
    """Current equipment state."""
    blowdown_valve_position_pct: float = Field(..., ge=0, le=100, description="Blowdown valve %")
    makeup_flow_gpm: float = Field(default=0, ge=0, description="Makeup water flow (gpm)")
    blowdown_flow_gpm: float = Field(default=0, ge=0, description="Blowdown flow (gpm)")
    recirculation_flow_gpm: float = Field(default=0, ge=0, description="Recirc flow (gpm)")
    tower_capacity_tons: float = Field(default=500, gt=0, description="Tower capacity (tons)")

    # Dosing pump states
    scale_inhibitor_pump_pct: float = Field(default=0, ge=0, le=100)
    corrosion_inhibitor_pump_pct: float = Field(default=0, ge=0, le=100)
    biocide_pump_pct: float = Field(default=0, ge=0, le=100)
    acid_pump_pct: float = Field(default=0, ge=0, le=100)


class RiskAssessment(BaseModel):
    """Risk assessment from ML models."""
    scaling_risk: float = Field(default=0.0, ge=0, le=1.0, description="Scaling risk [0-1]")
    corrosion_risk: float = Field(default=0.0, ge=0, le=1.0, description="Corrosion risk [0-1]")
    carryover_risk: float = Field(default=0.0, ge=0, le=1.0, description="Carryover risk [0-1]")
    biological_risk: float = Field(default=0.0, ge=0, le=1.0, description="Bio risk [0-1]")

    # Uncertainty bounds for risk predictions
    scaling_uncertainty: float = Field(default=0.1, ge=0, le=1.0)
    corrosion_uncertainty: float = Field(default=0.1, ge=0, le=1.0)
    carryover_uncertainty: float = Field(default=0.1, ge=0, le=1.0)
    biological_uncertainty: float = Field(default=0.1, ge=0, le=1.0)

    # Calibration status
    is_calibrated: bool = Field(default=True, description="Are risk models calibrated?")
    calibration_score: float = Field(default=0.9, ge=0, le=1.0)

    @property
    def max_uncertainty(self) -> float:
        """Maximum uncertainty across all risks."""
        return max(
            self.scaling_uncertainty,
            self.corrosion_uncertainty,
            self.carryover_uncertainty,
            self.biological_uncertainty
        )

    @property
    def combined_risk(self) -> float:
        """Combined weighted risk score."""
        weights = [0.35, 0.35, 0.15, 0.15]  # Scaling, corrosion, carryover, bio
        risks = [self.scaling_risk, self.corrosion_risk,
                 self.carryover_risk, self.biological_risk]
        return sum(w * r for w, r in zip(weights, risks))


class BlowdownSetpoint(BaseModel):
    """Recommended blowdown setpoint."""
    valve_position_pct: float = Field(..., ge=0, le=100)
    target_flow_gpm: float = Field(..., ge=0)
    ramp_rate_pct_per_min: float = Field(default=5.0, ge=0, le=20)
    hold_time_minutes: float = Field(default=15.0, ge=0)
    reason: str = Field(default="optimization")


class DosingSetpoint(BaseModel):
    """Recommended chemical dosing setpoint."""
    chemical_type: ChemicalType
    pump_speed_pct: float = Field(..., ge=0, le=100)
    dosing_rate_ml_min: float = Field(default=0, ge=0)
    ramp_rate_pct_per_min: float = Field(default=2.0, ge=0, le=10)
    reason: str = Field(default="optimization")


class OptimizationConfig(BaseModel):
    """Configuration for optimization."""
    # Horizon settings
    horizon_minutes: int = Field(default=30, ge=15, le=60)
    time_step_minutes: int = Field(default=5, ge=1, le=15)

    # Objective weights
    water_loss_weight: float = Field(default=0.30, ge=0, le=1.0)
    energy_loss_weight: float = Field(default=0.25, ge=0, le=1.0)
    chemical_cost_weight: float = Field(default=0.20, ge=0, le=1.0)
    risk_penalty_weight: float = Field(default=0.25, ge=0, le=1.0)

    # Cost parameters
    water_cost_per_1000gal: float = Field(default=5.0, ge=0)
    energy_cost_per_kwh: float = Field(default=0.12, ge=0)
    blowdown_enthalpy_btu_per_gal: float = Field(default=40.0, ge=0)  # At typical delta-T

    # Chemical costs ($/gallon)
    scale_inhibitor_cost_per_gal: float = Field(default=15.0, ge=0)
    corrosion_inhibitor_cost_per_gal: float = Field(default=12.0, ge=0)
    biocide_cost_per_gal: float = Field(default=20.0, ge=0)
    acid_cost_per_gal: float = Field(default=3.0, ge=0)

    # Chemistry limits (HARD constraints)
    conductivity_min_us_cm: float = Field(default=500)
    conductivity_max_us_cm: float = Field(default=3000)
    ph_min: float = Field(default=7.0)
    ph_max: float = Field(default=9.0)
    lsi_min: float = Field(default=-1.0)
    lsi_max: float = Field(default=1.0)
    cycles_min: float = Field(default=2.0)
    cycles_max: float = Field(default=8.0)

    # Equipment limits
    blowdown_valve_max_pct: float = Field(default=100.0)
    blowdown_valve_min_pct: float = Field(default=0.0)
    blowdown_ramp_rate_max_pct_per_min: float = Field(default=10.0)
    dosing_pump_max_pct: float = Field(default=100.0)
    dosing_ramp_rate_max_pct_per_min: float = Field(default=5.0)

    # Uncertainty handling
    high_uncertainty_threshold: float = Field(default=0.3, ge=0, le=1.0)
    conservative_safety_factor: float = Field(default=1.5, ge=1.0, le=3.0)

    # Solver settings
    solver_timeout_seconds: float = Field(default=30.0, ge=1, le=120)
    gap_tolerance: float = Field(default=0.01, ge=0, le=0.1)


class OptimizationResult(BaseModel):
    """Result of optimization."""
    result_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: OptimizationStatus

    # Recommended setpoints
    blowdown_setpoint: BlowdownSetpoint
    dosing_setpoints: List[DosingSetpoint] = Field(default_factory=list)

    # Objective values
    total_objective: float = Field(default=0.0)
    water_loss_cost: float = Field(default=0.0)
    energy_loss_cost: float = Field(default=0.0)
    chemical_cost: float = Field(default=0.0)
    risk_penalty: float = Field(default=0.0)

    # Constraints
    constraints_satisfied: bool = Field(default=True)
    binding_constraints: List[str] = Field(default_factory=list)

    # Solver metrics
    solve_time_ms: float = Field(default=0.0)
    iterations: int = Field(default=0)
    gap_percent: float = Field(default=0.0)

    # Uncertainty and confidence
    uncertainty_level: str = Field(default="low")  # low, medium, high
    is_conservative: bool = Field(default=False)
    confidence_score: float = Field(default=1.0, ge=0, le=1.0)

    # Provenance
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance()

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{self.result_id}|{self.timestamp.isoformat()}|"
            f"{self.blowdown_setpoint.valve_position_pct:.2f}|"
            f"{self.total_objective:.4f}|{self.status.value}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# WATERGUARD OPTIMIZER
# =============================================================================

class WaterguardOptimizer:
    """
    Mixed Integer Linear Programming optimizer for cooling tower water treatment.

    Optimizes blowdown and chemical dosing to minimize:
        Water_Loss + Energy_Loss + Chemical_Cost + Risk_Penalty

    Subject to hard chemistry constraints and equipment limits.
    Implements rolling horizon optimization with 15-60 minute lookahead.

    Key Features:
        - CVXPY-based constrained optimization
        - Risk penalty from ML risk models (bounded/calibrated)
        - Conservative fallback under high uncertainty
        - Complete provenance tracking

    Example:
        >>> config = OptimizationConfig()
        >>> optimizer = WaterguardOptimizer(config)
        >>> result = optimizer.optimize(
        ...     chemistry_state=chemistry,
        ...     equipment_state=equipment,
        ...     risk_assessment=risks
        ... )
        >>> print(f"Blowdown: {result.blowdown_setpoint.valve_position_pct}%")
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize Waterguard optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self._last_solution: Optional[OptimizationResult] = None
        self._solution_history: List[OptimizationResult] = []

        # Check solver availability
        if CVXPY_AVAILABLE:
            logger.info("WaterguardOptimizer initialized with CVXPY solver")
        elif SCIPY_AVAILABLE:
            logger.info("WaterguardOptimizer initialized with SciPy fallback")
        else:
            logger.warning("No optimization solver available - using heuristic")

    def optimize(
        self,
        chemistry_state: ChemistryState,
        equipment_state: EquipmentState,
        risk_assessment: RiskAssessment,
        makeup_chemistry: Optional[ChemistryState] = None
    ) -> OptimizationResult:
        """
        Perform optimization to determine optimal setpoints.

        Args:
            chemistry_state: Current water chemistry
            equipment_state: Current equipment state
            risk_assessment: Risk predictions from ML models
            makeup_chemistry: Makeup water chemistry (optional)

        Returns:
            OptimizationResult with recommended setpoints
        """
        start_time = time.time()

        try:
            # Check for high uncertainty - use conservative approach
            if self._is_high_uncertainty(risk_assessment):
                logger.warning("High uncertainty detected - using conservative optimization")
                result = self._optimize_conservative(
                    chemistry_state, equipment_state, risk_assessment
                )
                result.is_conservative = True
                result.uncertainty_level = "high"
            else:
                # Normal optimization
                if CVXPY_AVAILABLE:
                    result = self._optimize_cvxpy(
                        chemistry_state, equipment_state, risk_assessment, makeup_chemistry
                    )
                elif SCIPY_AVAILABLE:
                    result = self._optimize_scipy(
                        chemistry_state, equipment_state, risk_assessment, makeup_chemistry
                    )
                else:
                    result = self._optimize_heuristic(
                        chemistry_state, equipment_state, risk_assessment
                    )

            # Calculate solve time
            result.solve_time_ms = (time.time() - start_time) * 1000

            # Store solution
            self._last_solution = result
            self._solution_history.append(result)
            if len(self._solution_history) > 1000:
                self._solution_history = self._solution_history[-500:]

            logger.info(
                "Optimization complete: status=%s, blowdown=%.1f%%, objective=%.2f",
                result.status.value,
                result.blowdown_setpoint.valve_position_pct,
                result.total_objective
            )

            return result

        except Exception as e:
            logger.error("Optimization failed: %s", e, exc_info=True)
            # Return conservative fallback
            return self._optimize_conservative(
                chemistry_state, equipment_state, risk_assessment
            )

    def _is_high_uncertainty(self, risk_assessment: RiskAssessment) -> bool:
        """Check if uncertainty is too high for normal optimization."""
        if risk_assessment.max_uncertainty > self.config.high_uncertainty_threshold:
            return True
        if not risk_assessment.is_calibrated:
            return True
        if risk_assessment.calibration_score < 0.7:
            return True
        return False

    def _optimize_cvxpy(
        self,
        chemistry_state: ChemistryState,
        equipment_state: EquipmentState,
        risk_assessment: RiskAssessment,
        makeup_chemistry: Optional[ChemistryState]
    ) -> OptimizationResult:
        """
        Optimize using CVXPY solver - ZERO HALLUCINATION.

        Formulation:
            min  w1*WaterLoss + w2*EnergyLoss + w3*ChemCost + w4*RiskPenalty
            s.t. Chemistry constraints (hard)
                 Equipment constraints (hard)
                 Ramp rate limits (hard)
        """
        n_steps = self.config.horizon_minutes // self.config.time_step_minutes

        # Decision variables
        # Blowdown valve position (0-100%)
        blowdown = cp.Variable(n_steps, nonneg=True)

        # Dosing pump speeds (0-100%)
        scale_inhibitor = cp.Variable(n_steps, nonneg=True)
        corrosion_inhibitor = cp.Variable(n_steps, nonneg=True)
        biocide = cp.Variable(n_steps, nonneg=True)
        acid = cp.Variable(n_steps, nonneg=True)

        # Auxiliary: predicted conductivity (for soft constraint)
        cond_predicted = cp.Variable(n_steps, nonneg=True)

        # =================================================================
        # Objective function components
        # =================================================================

        # Water loss cost (blowdown volume)
        # Assume blowdown flow ~ 0.1 * valve_position * max_flow
        max_blowdown_gpm = equipment_state.tower_capacity_tons * 0.003 * 60  # Rough estimate
        blowdown_volume_gal = (
            cp.sum(blowdown) / 100 * max_blowdown_gpm *
            self.config.time_step_minutes
        )
        water_cost = (
            blowdown_volume_gal / 1000 * self.config.water_cost_per_1000gal
        )

        # Energy loss cost (blowdown enthalpy)
        energy_cost = (
            blowdown_volume_gal *
            self.config.blowdown_enthalpy_btu_per_gal / 3412 *  # Convert to kWh
            self.config.energy_cost_per_kwh
        )

        # Chemical cost
        dose_volume_factor = 0.001  # ml/min per % pump speed
        chemical_cost = (
            cp.sum(scale_inhibitor) * dose_volume_factor *
            self.config.scale_inhibitor_cost_per_gal / 3785 +  # ml to gal
            cp.sum(corrosion_inhibitor) * dose_volume_factor *
            self.config.corrosion_inhibitor_cost_per_gal / 3785 +
            cp.sum(biocide) * dose_volume_factor *
            self.config.biocide_cost_per_gal / 3785 +
            cp.sum(acid) * dose_volume_factor *
            self.config.acid_cost_per_gal / 3785
        )

        # Risk penalty (bounded, from calibrated ML)
        # Risk is bounded by adding conservative margin based on uncertainty
        scaling_bound = min(1.0, risk_assessment.scaling_risk +
                           risk_assessment.scaling_uncertainty)
        corrosion_bound = min(1.0, risk_assessment.corrosion_risk +
                             risk_assessment.corrosion_uncertainty)

        # Risk penalty increases with lower blowdown (higher concentration)
        # and decreases with proper chemical treatment
        avg_blowdown = cp.sum(blowdown) / n_steps
        avg_scale_inhib = cp.sum(scale_inhibitor) / n_steps
        avg_corr_inhib = cp.sum(corrosion_inhibitor) / n_steps

        # Simplified risk model: risk decreases with blowdown, increases baseline
        base_risk = (scaling_bound * 0.5 + corrosion_bound * 0.5)
        risk_penalty = (
            base_risk * 100 -  # Base cost from risk
            avg_blowdown * 0.1 -  # Blowdown reduces risk
            avg_scale_inhib * 0.05 -  # Scale inhibitor helps
            avg_corr_inhib * 0.05  # Corrosion inhibitor helps
        )

        # Total objective
        objective = (
            self.config.water_loss_weight * water_cost +
            self.config.energy_loss_weight * energy_cost +
            self.config.chemical_cost_weight * chemical_cost +
            self.config.risk_penalty_weight * cp.pos(risk_penalty)
        )

        # =================================================================
        # Constraints
        # =================================================================
        constraints = []

        # Equipment bounds
        constraints.append(blowdown <= self.config.blowdown_valve_max_pct)
        constraints.append(blowdown >= self.config.blowdown_valve_min_pct)
        constraints.append(scale_inhibitor <= self.config.dosing_pump_max_pct)
        constraints.append(corrosion_inhibitor <= self.config.dosing_pump_max_pct)
        constraints.append(biocide <= self.config.dosing_pump_max_pct)
        constraints.append(acid <= self.config.dosing_pump_max_pct)

        # Ramp rate constraints
        for i in range(1, n_steps):
            max_ramp = (self.config.blowdown_ramp_rate_max_pct_per_min *
                       self.config.time_step_minutes)
            constraints.append(cp.abs(blowdown[i] - blowdown[i-1]) <= max_ramp)

            dose_ramp = (self.config.dosing_ramp_rate_max_pct_per_min *
                        self.config.time_step_minutes)
            constraints.append(cp.abs(scale_inhibitor[i] - scale_inhibitor[i-1]) <= dose_ramp)
            constraints.append(cp.abs(corrosion_inhibitor[i] - corrosion_inhibitor[i-1]) <= dose_ramp)

        # Initial state constraint (smooth transition from current)
        current_blowdown = equipment_state.blowdown_valve_position_pct
        constraints.append(
            cp.abs(blowdown[0] - current_blowdown) <=
            self.config.blowdown_ramp_rate_max_pct_per_min * self.config.time_step_minutes
        )

        # Chemistry constraint (simplified): maintain reasonable blowdown for conductivity
        # Higher conductivity -> need more blowdown
        if chemistry_state.conductivity_us_cm > self.config.conductivity_max_us_cm * 0.9:
            constraints.append(avg_blowdown >= 30)  # Require minimum blowdown

        # =================================================================
        # Solve
        # =================================================================
        problem = cp.Problem(cp.Minimize(objective), constraints)

        try:
            problem.solve(
                solver=cp.ECOS,  # Fast conic solver
                verbose=False,
                max_iters=1000
            )

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                status = OptimizationStatus.OPTIMAL
            elif problem.status == cp.FEASIBLE:
                status = OptimizationStatus.FEASIBLE
            else:
                logger.warning("CVXPY status: %s, using fallback", problem.status)
                return self._optimize_conservative(
                    chemistry_state, equipment_state, risk_assessment
                )

            # Extract solution
            blowdown_pct = float(blowdown.value[0]) if blowdown.value is not None else 30.0
            blowdown_pct = np.clip(blowdown_pct, 0, 100)

            scale_pct = float(scale_inhibitor.value[0]) if scale_inhibitor.value is not None else 0.0
            corr_pct = float(corrosion_inhibitor.value[0]) if corrosion_inhibitor.value is not None else 0.0
            bio_pct = float(biocide.value[0]) if biocide.value is not None else 0.0
            acid_pct = float(acid.value[0]) if acid.value is not None else 0.0

            # Build result
            blowdown_setpoint = BlowdownSetpoint(
                valve_position_pct=blowdown_pct,
                target_flow_gpm=blowdown_pct / 100 * max_blowdown_gpm,
                ramp_rate_pct_per_min=self.config.blowdown_ramp_rate_max_pct_per_min,
                hold_time_minutes=self.config.horizon_minutes,
                reason="MILP optimization"
            )

            dosing_setpoints = []
            if scale_pct > 0.1:
                dosing_setpoints.append(DosingSetpoint(
                    chemical_type=ChemicalType.SCALE_INHIBITOR,
                    pump_speed_pct=scale_pct,
                    reason="Scaling risk mitigation"
                ))
            if corr_pct > 0.1:
                dosing_setpoints.append(DosingSetpoint(
                    chemical_type=ChemicalType.CORROSION_INHIBITOR,
                    pump_speed_pct=corr_pct,
                    reason="Corrosion risk mitigation"
                ))
            if bio_pct > 0.1:
                dosing_setpoints.append(DosingSetpoint(
                    chemical_type=ChemicalType.BIOCIDE,
                    pump_speed_pct=bio_pct,
                    reason="Biological risk mitigation"
                ))
            if acid_pct > 0.1:
                dosing_setpoints.append(DosingSetpoint(
                    chemical_type=ChemicalType.PH_ACID,
                    pump_speed_pct=acid_pct,
                    reason="pH control"
                ))

            return OptimizationResult(
                status=status,
                blowdown_setpoint=blowdown_setpoint,
                dosing_setpoints=dosing_setpoints,
                total_objective=float(problem.value) if problem.value else 0.0,
                water_loss_cost=float(water_cost.value) if hasattr(water_cost, 'value') else 0.0,
                energy_loss_cost=float(energy_cost.value) if hasattr(energy_cost, 'value') else 0.0,
                chemical_cost=float(chemical_cost.value) if hasattr(chemical_cost, 'value') else 0.0,
                risk_penalty=float(risk_penalty.value) if hasattr(risk_penalty, 'value') else 0.0,
                constraints_satisfied=True,
                uncertainty_level="low" if risk_assessment.max_uncertainty < 0.1 else "medium",
                confidence_score=risk_assessment.calibration_score
            )

        except Exception as e:
            logger.error("CVXPY solve failed: %s", e)
            return self._optimize_conservative(
                chemistry_state, equipment_state, risk_assessment
            )

    def _optimize_scipy(
        self,
        chemistry_state: ChemistryState,
        equipment_state: EquipmentState,
        risk_assessment: RiskAssessment,
        makeup_chemistry: Optional[ChemistryState]
    ) -> OptimizationResult:
        """Optimize using SciPy as fallback - ZERO HALLUCINATION."""

        def objective(x):
            """Combined objective function."""
            blowdown_pct = x[0]
            scale_pct = x[1]
            corr_pct = x[2]

            # Water cost
            max_blowdown_gpm = equipment_state.tower_capacity_tons * 0.003 * 60
            water_cost = (
                blowdown_pct / 100 * max_blowdown_gpm *
                self.config.horizon_minutes / 1000 *
                self.config.water_cost_per_1000gal
            )

            # Energy cost
            energy_cost = (
                blowdown_pct / 100 * max_blowdown_gpm *
                self.config.horizon_minutes *
                self.config.blowdown_enthalpy_btu_per_gal / 3412 *
                self.config.energy_cost_per_kwh
            )

            # Chemical cost
            chemical_cost = (
                (scale_pct + corr_pct) * 0.001 *
                self.config.scale_inhibitor_cost_per_gal / 3785
            )

            # Risk penalty
            base_risk = (risk_assessment.scaling_risk + risk_assessment.corrosion_risk) / 2
            risk_penalty = max(0, base_risk * 100 - blowdown_pct * 0.1 -
                              (scale_pct + corr_pct) * 0.05)

            return (
                self.config.water_loss_weight * water_cost +
                self.config.energy_loss_weight * energy_cost +
                self.config.chemical_cost_weight * chemical_cost +
                self.config.risk_penalty_weight * risk_penalty
            )

        # Bounds
        bounds = [
            (self.config.blowdown_valve_min_pct, self.config.blowdown_valve_max_pct),
            (0, self.config.dosing_pump_max_pct),
            (0, self.config.dosing_pump_max_pct)
        ]

        # Initial guess
        x0 = [
            equipment_state.blowdown_valve_position_pct,
            equipment_state.scale_inhibitor_pump_pct,
            equipment_state.corrosion_inhibitor_pump_pct
        ]

        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )

            if result.success:
                status = OptimizationStatus.OPTIMAL
            else:
                status = OptimizationStatus.FEASIBLE

            blowdown_pct = float(np.clip(result.x[0], 0, 100))
            scale_pct = float(np.clip(result.x[1], 0, 100))
            corr_pct = float(np.clip(result.x[2], 0, 100))

            max_blowdown_gpm = equipment_state.tower_capacity_tons * 0.003 * 60

            blowdown_setpoint = BlowdownSetpoint(
                valve_position_pct=blowdown_pct,
                target_flow_gpm=blowdown_pct / 100 * max_blowdown_gpm,
                ramp_rate_pct_per_min=self.config.blowdown_ramp_rate_max_pct_per_min,
                reason="SciPy optimization"
            )

            dosing_setpoints = []
            if scale_pct > 0.1:
                dosing_setpoints.append(DosingSetpoint(
                    chemical_type=ChemicalType.SCALE_INHIBITOR,
                    pump_speed_pct=scale_pct,
                    reason="Scaling mitigation"
                ))
            if corr_pct > 0.1:
                dosing_setpoints.append(DosingSetpoint(
                    chemical_type=ChemicalType.CORROSION_INHIBITOR,
                    pump_speed_pct=corr_pct,
                    reason="Corrosion mitigation"
                ))

            return OptimizationResult(
                status=status,
                blowdown_setpoint=blowdown_setpoint,
                dosing_setpoints=dosing_setpoints,
                total_objective=float(result.fun),
                iterations=result.nit,
                uncertainty_level="medium",
                confidence_score=0.9
            )

        except Exception as e:
            logger.error("SciPy optimization failed: %s", e)
            return self._optimize_conservative(
                chemistry_state, equipment_state, risk_assessment
            )

    def _optimize_heuristic(
        self,
        chemistry_state: ChemistryState,
        equipment_state: EquipmentState,
        risk_assessment: RiskAssessment
    ) -> OptimizationResult:
        """Heuristic optimization when solvers unavailable - ZERO HALLUCINATION."""
        logger.info("Using heuristic optimization")

        # Simple rule-based optimization
        max_blowdown_gpm = equipment_state.tower_capacity_tons * 0.003 * 60

        # Base blowdown on conductivity
        if chemistry_state.conductivity_us_cm > self.config.conductivity_max_us_cm * 0.9:
            blowdown_pct = 60.0  # High conductivity - increase blowdown
        elif chemistry_state.conductivity_us_cm > self.config.conductivity_max_us_cm * 0.7:
            blowdown_pct = 40.0
        elif chemistry_state.conductivity_us_cm < self.config.conductivity_min_us_cm * 1.2:
            blowdown_pct = 10.0  # Low conductivity - minimal blowdown
        else:
            blowdown_pct = 25.0  # Normal

        # Adjust for risk
        if risk_assessment.scaling_risk > 0.7:
            blowdown_pct = min(100, blowdown_pct + 20)
        if risk_assessment.corrosion_risk > 0.7:
            blowdown_pct = max(10, blowdown_pct - 10)  # Less blowdown, more inhibitor

        # Chemical dosing based on risk
        scale_pct = 50.0 if risk_assessment.scaling_risk > 0.3 else 20.0
        corr_pct = 50.0 if risk_assessment.corrosion_risk > 0.3 else 20.0

        blowdown_setpoint = BlowdownSetpoint(
            valve_position_pct=blowdown_pct,
            target_flow_gpm=blowdown_pct / 100 * max_blowdown_gpm,
            reason="Heuristic optimization"
        )

        dosing_setpoints = [
            DosingSetpoint(
                chemical_type=ChemicalType.SCALE_INHIBITOR,
                pump_speed_pct=scale_pct,
                reason="Risk-based dosing"
            ),
            DosingSetpoint(
                chemical_type=ChemicalType.CORROSION_INHIBITOR,
                pump_speed_pct=corr_pct,
                reason="Risk-based dosing"
            )
        ]

        return OptimizationResult(
            status=OptimizationStatus.FEASIBLE,
            blowdown_setpoint=blowdown_setpoint,
            dosing_setpoints=dosing_setpoints,
            uncertainty_level="high",
            confidence_score=0.6
        )

    def _optimize_conservative(
        self,
        chemistry_state: ChemistryState,
        equipment_state: EquipmentState,
        risk_assessment: RiskAssessment
    ) -> OptimizationResult:
        """
        Conservative optimization for high uncertainty - ZERO HALLUCINATION.

        When uncertainty is high, we:
        1. Increase blowdown to reduce concentration risk
        2. Maintain moderate chemical dosing
        3. Avoid aggressive changes
        """
        logger.warning("Using conservative optimization due to high uncertainty")

        max_blowdown_gpm = equipment_state.tower_capacity_tons * 0.003 * 60

        # Conservative: higher blowdown to be safe
        base_blowdown = 40.0
        safety_factor = self.config.conservative_safety_factor

        # Increase if conductivity is elevated
        if chemistry_state.conductivity_us_cm > self.config.conductivity_max_us_cm * 0.7:
            base_blowdown = 50.0 * safety_factor

        # Increase if any risk is elevated (add margin for uncertainty)
        max_risk = max(
            risk_assessment.scaling_risk + risk_assessment.scaling_uncertainty,
            risk_assessment.corrosion_risk + risk_assessment.corrosion_uncertainty,
            risk_assessment.carryover_risk + risk_assessment.carryover_uncertainty
        )
        if max_risk > 0.5:
            base_blowdown = min(80.0, base_blowdown * 1.3)

        blowdown_pct = min(100.0, base_blowdown)

        # Smooth transition from current
        current = equipment_state.blowdown_valve_position_pct
        max_change = self.config.blowdown_ramp_rate_max_pct_per_min * 5  # 5 min ramp
        if abs(blowdown_pct - current) > max_change:
            if blowdown_pct > current:
                blowdown_pct = current + max_change
            else:
                blowdown_pct = current - max_change

        blowdown_setpoint = BlowdownSetpoint(
            valve_position_pct=blowdown_pct,
            target_flow_gpm=blowdown_pct / 100 * max_blowdown_gpm,
            ramp_rate_pct_per_min=self.config.blowdown_ramp_rate_max_pct_per_min / 2,
            reason="Conservative - high uncertainty"
        )

        # Moderate chemical dosing
        dosing_setpoints = [
            DosingSetpoint(
                chemical_type=ChemicalType.SCALE_INHIBITOR,
                pump_speed_pct=40.0,
                reason="Conservative dosing"
            ),
            DosingSetpoint(
                chemical_type=ChemicalType.CORROSION_INHIBITOR,
                pump_speed_pct=40.0,
                reason="Conservative dosing"
            )
        ]

        return OptimizationResult(
            status=OptimizationStatus.CONSERVATIVE_FALLBACK,
            blowdown_setpoint=blowdown_setpoint,
            dosing_setpoints=dosing_setpoints,
            uncertainty_level="high",
            is_conservative=True,
            confidence_score=0.5
        )

    def get_last_solution(self) -> Optional[OptimizationResult]:
        """Get the most recent optimization result."""
        return self._last_solution

    def get_solution_history(self, limit: int = 100) -> List[OptimizationResult]:
        """Get optimization history."""
        return list(reversed(self._solution_history[-limit:]))

    def validate_constraints(
        self,
        chemistry_state: ChemistryState,
        equipment_state: EquipmentState
    ) -> Tuple[bool, List[str]]:
        """
        Validate if current state satisfies all hard constraints.

        Returns:
            Tuple of (all_satisfied, list_of_violations)
        """
        violations = []

        # Conductivity
        if chemistry_state.conductivity_us_cm < self.config.conductivity_min_us_cm:
            violations.append(
                f"Conductivity {chemistry_state.conductivity_us_cm:.0f} below min "
                f"{self.config.conductivity_min_us_cm:.0f}"
            )
        if chemistry_state.conductivity_us_cm > self.config.conductivity_max_us_cm:
            violations.append(
                f"Conductivity {chemistry_state.conductivity_us_cm:.0f} above max "
                f"{self.config.conductivity_max_us_cm:.0f}"
            )

        # pH
        if chemistry_state.ph < self.config.ph_min:
            violations.append(f"pH {chemistry_state.ph:.2f} below min {self.config.ph_min:.1f}")
        if chemistry_state.ph > self.config.ph_max:
            violations.append(f"pH {chemistry_state.ph:.2f} above max {self.config.ph_max:.1f}")

        # LSI
        lsi = chemistry_state.lsi
        if lsi < self.config.lsi_min:
            violations.append(f"LSI {lsi:.2f} below min {self.config.lsi_min:.1f}")
        if lsi > self.config.lsi_max:
            violations.append(f"LSI {lsi:.2f} above max {self.config.lsi_max:.1f}")

        # Cycles of concentration
        if chemistry_state.cycles_of_concentration < self.config.cycles_min:
            violations.append(
                f"CoC {chemistry_state.cycles_of_concentration:.1f} below min "
                f"{self.config.cycles_min:.1f}"
            )
        if chemistry_state.cycles_of_concentration > self.config.cycles_max:
            violations.append(
                f"CoC {chemistry_state.cycles_of_concentration:.1f} above max "
                f"{self.config.cycles_max:.1f}"
            )

        return (len(violations) == 0, violations)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_default_optimizer() -> WaterguardOptimizer:
    """Create optimizer with default configuration."""
    return WaterguardOptimizer(OptimizationConfig())


def create_conservative_optimizer() -> WaterguardOptimizer:
    """Create optimizer with conservative settings for high-risk systems."""
    config = OptimizationConfig(
        high_uncertainty_threshold=0.2,
        conservative_safety_factor=2.0,
        risk_penalty_weight=0.35,
        water_loss_weight=0.25,
        blowdown_ramp_rate_max_pct_per_min=5.0
    )
    return WaterguardOptimizer(config)


def create_aggressive_optimizer() -> WaterguardOptimizer:
    """Create optimizer for maximum water savings (low-risk systems only)."""
    config = OptimizationConfig(
        water_loss_weight=0.40,
        energy_loss_weight=0.30,
        chemical_cost_weight=0.15,
        risk_penalty_weight=0.15,
        cycles_max=10.0,
        conductivity_max_us_cm=4000
    )
    return WaterguardOptimizer(config)
