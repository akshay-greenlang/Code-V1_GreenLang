# -*- coding: utf-8 -*-
"""
Repair Prioritization Engine for GL-015 INSULSCAN.

Intelligent repair prioritization system for insulation maintenance including:
- Multi-factor criticality scoring
- ROI-based ranking with NPV analysis
- Risk-based prioritization (FMEA methodology)
- Schedule optimization with resource leveling
- Budget-constrained optimization (knapsack problem)
- Work order generation with material/labor estimates

Zero-hallucination design: All calculations are deterministic using
standard engineering economics and risk assessment methodologies.

References:
- ASTM C1055 Standard Guide for Heated System Surface Conditions
- ASTM C680 Standard Practice for Heat Flux Calculation
- CINI Manual for Thermal Insulation
- ISO 12241 Thermal Insulation for Building Equipment
- NACE SP0198 Control of Corrosion Under Insulation
- MIL-STD-1629A Failure Mode Effects and Criticality Analysis
- IEC 60812 FMEA/FMECA Analysis Techniques

Author: GreenLang AI Agent Factory - GL-CalculatorEngineer
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union, FrozenSet, Set
from datetime import datetime, date, timedelta
from collections import defaultdict
from functools import lru_cache

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Reference Data
# =============================================================================

# Heat loss severity thresholds (W/m for pipe, W/m2 for flat surfaces)
HEAT_LOSS_SEVERITY_THRESHOLDS: Dict[str, Tuple[float, float, float, float]] = {
    # (minor, moderate, significant, severe)
    'hot_pipe_uninsulated': (50.0, 150.0, 300.0, 500.0),
    'hot_pipe_damaged': (25.0, 75.0, 150.0, 250.0),
    'hot_flat_surface': (100.0, 250.0, 500.0, 1000.0),
    'cold_pipe_uninsulated': (20.0, 50.0, 100.0, 200.0),
    'cold_pipe_damaged': (10.0, 30.0, 60.0, 120.0),
    'cold_flat_surface': (50.0, 125.0, 250.0, 500.0),
}

# Personnel safety temperature limits (ASTM C1055)
PERSONNEL_SAFETY_TEMP_C: Dict[str, float] = {
    'momentary_contact': 60.0,  # 5-second contact
    'brief_contact': 55.0,      # 10-second contact
    'continuous_contact': 48.0,  # Extended contact
    'foot_contact': 44.0,        # Standing surface
    'cold_burn_threshold': -20.0,  # Cryogenic exposure
}

# Insulation material cost per linear meter (typical installed)
INSULATION_COST_PER_METER: Dict[str, Dict[str, float]] = {
    # {material: {thickness_mm: cost_usd}}
    'mineral_wool': {25: 15.0, 50: 22.0, 75: 30.0, 100: 40.0, 150: 55.0},
    'calcium_silicate': {25: 25.0, 50: 38.0, 75: 52.0, 100: 68.0, 150: 95.0},
    'cellular_glass': {25: 35.0, 50: 55.0, 75: 78.0, 100: 105.0, 150: 150.0},
    'perlite': {25: 22.0, 50: 35.0, 75: 48.0, 100: 62.0, 150: 88.0},
    'aerogel': {10: 120.0, 15: 175.0, 20: 240.0, 25: 310.0},
    'polyurethane_foam': {25: 18.0, 50: 28.0, 75: 40.0, 100: 52.0},
    'polystyrene': {25: 12.0, 50: 18.0, 75: 25.0, 100: 32.0},
    'elastomeric': {13: 22.0, 19: 30.0, 25: 40.0, 32: 52.0},
}

# Labor rates for insulation work (USD per hour)
LABOR_RATES: Dict[str, float] = {
    'journeyman_insulator': 85.0,
    'apprentice_insulator': 55.0,
    'supervisor': 110.0,
    'scaffolding_worker': 75.0,
    'painter_jacket': 70.0,
}

# Production rates (linear meters per hour per worker)
PRODUCTION_RATES: Dict[str, float] = {
    'removal_damaged': 8.0,
    'surface_preparation': 12.0,
    'insulation_installation_simple': 6.0,
    'insulation_installation_complex': 3.0,
    'jacketing_aluminum': 10.0,
    'jacketing_stainless': 7.0,
    'painting': 25.0,
    'inspection': 40.0,
}

# Corrosion Under Insulation (CUI) risk factors
CUI_RISK_FACTORS: Dict[str, Dict[str, float]] = {
    'temperature_range': {
        '0_50C': 0.3,      # Low risk
        '50_150C': 1.0,    # High risk (wet-dry cycling)
        '150_250C': 0.6,   # Moderate risk
        'above_250C': 0.2,  # Low risk (too hot for moisture)
        'below_0C': 0.4,    # Ice formation risk
    },
    'environment': {
        'indoor_dry': 0.2,
        'indoor_humid': 0.5,
        'outdoor_temperate': 0.6,
        'outdoor_marine': 1.0,
        'outdoor_industrial': 0.9,
        'outdoor_desert': 0.3,
    },
    'insulation_condition': {
        'excellent': 0.1,
        'good': 0.3,
        'fair': 0.6,
        'poor': 0.9,
        'failed': 1.0,
    },
}


# =============================================================================
# Enumerations
# =============================================================================

class PriorityCategory(str, Enum):
    """Repair priority categories with time-to-action."""
    EMERGENCY = "emergency"      # Immediate action required
    URGENT = "urgent"            # Within 30 days
    HIGH = "high"                # Next turnaround
    MEDIUM = "medium"            # Within 12 months
    LOW = "low"                  # Routine maintenance
    MONITOR = "monitor"          # Reinspect, no action


class RiskLevel(str, Enum):
    """Risk matrix classification."""
    EXTREME = "extreme"    # Unacceptable, immediate mitigation
    HIGH = "high"          # Significant, mitigation required
    MEDIUM = "medium"      # Moderate, management attention
    LOW = "low"            # Acceptable, routine monitoring
    NEGLIGIBLE = "negligible"  # No action required


class InsulationMaterial(str, Enum):
    """Insulation material types."""
    MINERAL_WOOL = "mineral_wool"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    PERLITE = "perlite"
    AEROGEL = "aerogel"
    POLYURETHANE_FOAM = "polyurethane_foam"
    POLYSTYRENE = "polystyrene"
    ELASTOMERIC = "elastomeric"


class DamageType(str, Enum):
    """Types of insulation damage."""
    MISSING = "missing"
    COMPRESSED = "compressed"
    WET = "wet"
    CRACKED = "cracked"
    JACKET_DAMAGED = "jacket_damaged"
    VAPOR_BARRIER_FAILED = "vapor_barrier_failed"
    THERMAL_BRIDGING = "thermal_bridging"
    CUI_SUSPECTED = "cui_suspected"


class RepairScope(str, Enum):
    """Scope of repair work."""
    SPOT_REPAIR = "spot_repair"        # < 1m
    SECTION_REPAIR = "section_repair"   # 1-10m
    ZONE_REPLACEMENT = "zone_replacement"  # 10-50m
    FULL_REPLACEMENT = "full_replacement"  # > 50m


class EquipmentType(str, Enum):
    """Equipment types for repair context."""
    PIPE = "pipe"
    VESSEL = "vessel"
    TANK = "tank"
    VALVE = "valve"
    FLANGE = "flange"
    HEAT_EXCHANGER = "heat_exchanger"
    DUCT = "duct"
    EQUIPMENT_MISC = "equipment_misc"


class OutageType(str, Enum):
    """Outage/turnaround types."""
    RUNNING = "running"           # Online repair possible
    SHUTDOWN = "shutdown"         # Unit shutdown required
    TURNAROUND = "turnaround"     # Major maintenance event
    EMERGENCY = "emergency_outage"  # Unplanned shutdown


# =============================================================================
# Severity Rating Scale (for FMEA)
# =============================================================================

class SeverityRating(IntEnum):
    """FMEA Severity ratings (1-10 scale per MIL-STD-1629A)."""
    NONE = 1           # No effect
    VERY_MINOR = 2     # Very slight effect
    MINOR = 3          # Minor effect
    MODERATE_LOW = 4   # Moderate effect
    MODERATE = 5       # Moderate effect with some impact
    MODERATE_HIGH = 6  # Significant effect
    HIGH = 7           # High impact
    VERY_HIGH = 8      # Very high impact
    CRITICAL = 9       # Critical with regulatory impact
    HAZARDOUS = 10     # Hazardous without warning


class OccurrenceRating(IntEnum):
    """FMEA Occurrence/Probability ratings (1-10 scale)."""
    REMOTE = 1         # < 1 in 1,000,000
    VERY_LOW = 2       # 1 in 100,000
    LOW = 3            # 1 in 10,000
    MODERATE_LOW = 4   # 1 in 2,000
    MODERATE = 5       # 1 in 500
    MODERATE_HIGH = 6  # 1 in 100
    HIGH = 7           # 1 in 50
    VERY_HIGH = 8      # 1 in 20
    ALMOST_CERTAIN = 9 # 1 in 10
    INEVITABLE = 10    # > 1 in 2


class DetectionRating(IntEnum):
    """FMEA Detection ratings (1-10 scale, higher = harder to detect)."""
    ALMOST_CERTAIN = 1  # Will definitely detect
    HIGH = 2            # High probability of detection
    MODERATELY_HIGH = 3
    MODERATE = 4
    MODERATELY_LOW = 5
    LOW = 6
    VERY_LOW = 7
    REMOTE = 8
    VERY_REMOTE = 9
    IMPOSSIBLE = 10     # Cannot detect


# =============================================================================
# Input Data Classes (Immutable)
# =============================================================================

@dataclass(frozen=True)
class DefectLocation:
    """Location information for a defect."""
    equipment_tag: str
    equipment_type: EquipmentType
    area_code: str
    unit_code: str
    elevation_m: float = 0.0
    access_difficulty: int = 1  # 1=easy, 5=difficult
    scaffold_required: bool = False
    permit_required: bool = False
    process_isolation_required: bool = False


@dataclass(frozen=True)
class ThermalDefect:
    """Thermal defect data from inspection."""
    defect_id: str
    location: DefectLocation
    damage_type: DamageType
    length_m: Decimal
    width_m: Optional[Decimal] = None
    process_temperature_c: Decimal = Decimal("100.0")
    surface_temperature_c: Decimal = Decimal("50.0")
    ambient_temperature_c: Decimal = Decimal("25.0")
    heat_loss_w_per_m: Decimal = Decimal("0.0")
    heat_loss_w_per_m2: Decimal = Decimal("0.0")
    insulation_material: InsulationMaterial = InsulationMaterial.MINERAL_WOOL
    insulation_thickness_mm: Decimal = Decimal("50.0")
    pipe_diameter_mm: Optional[Decimal] = None
    existing_condition_score: int = 5  # 1=new, 10=failed
    inspection_date: date = field(default_factory=date.today)
    inspector_id: str = ""
    thermal_image_id: str = ""
    notes: str = ""


@dataclass(frozen=True)
class CriticalityWeights:
    """Weighting factors for criticality scoring."""
    heat_loss_weight: Decimal = Decimal("0.25")
    safety_risk_weight: Decimal = Decimal("0.25")
    process_impact_weight: Decimal = Decimal("0.20")
    environmental_weight: Decimal = Decimal("0.15")
    asset_protection_weight: Decimal = Decimal("0.15")

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0."""
        total = (
            self.heat_loss_weight +
            self.safety_risk_weight +
            self.process_impact_weight +
            self.environmental_weight +
            self.asset_protection_weight
        )
        if abs(total - Decimal("1.0")) > Decimal("0.001"):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class EconomicParameters:
    """Economic parameters for ROI calculations."""
    energy_cost_per_kwh: Decimal = Decimal("0.12")
    operating_hours_per_year: Decimal = Decimal("8000")
    discount_rate_percent: Decimal = Decimal("8.0")
    equipment_life_years: int = 15
    inflation_rate_percent: Decimal = Decimal("2.5")
    carbon_price_per_tonne: Decimal = Decimal("50.0")
    co2_emission_factor_kg_per_kwh: Decimal = Decimal("0.185")
    labor_overhead_percent: Decimal = Decimal("35.0")
    material_markup_percent: Decimal = Decimal("15.0")


@dataclass(frozen=True)
class ScheduleConstraints:
    """Constraints for schedule optimization."""
    next_turnaround_date: Optional[date] = None
    turnaround_duration_days: int = 30
    max_concurrent_repairs: int = 5
    available_labor_hours_per_day: Decimal = Decimal("80.0")
    mobilization_cost_per_location: Decimal = Decimal("5000.0")
    minimum_batch_size_m: Decimal = Decimal("10.0")
    weather_restricted: bool = False
    shift_hours: int = 10


@dataclass(frozen=True)
class BudgetConstraint:
    """Budget constraint parameters."""
    total_budget: Decimal
    emergency_reserve_percent: Decimal = Decimal("15.0")
    phase_count: int = 1
    fiscal_year_end: Optional[date] = None


# =============================================================================
# Output Data Classes
# =============================================================================

@dataclass
class CriticalityScore:
    """Multi-factor criticality score result."""
    defect_id: str
    heat_loss_score: Decimal
    safety_risk_score: Decimal
    process_impact_score: Decimal
    environmental_score: Decimal
    asset_protection_score: Decimal
    composite_score: Decimal
    dominant_factor: str
    calculation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairROI:
    """ROI calculation result for a repair."""
    defect_id: str
    estimated_repair_cost: Decimal
    annual_energy_savings: Decimal
    annual_carbon_savings_tonnes: Decimal
    simple_payback_years: Decimal
    npv_over_life: Decimal
    irr_percent: Optional[Decimal]
    roi_percent: Decimal
    cost_per_unit_saved: Decimal  # Cost per kWh saved over life
    calculation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """FMEA-style risk assessment result."""
    defect_id: str
    severity_rating: SeverityRating
    occurrence_rating: OccurrenceRating
    detection_rating: DetectionRating
    risk_priority_number: int
    risk_level: RiskLevel
    failure_modes: List[str]
    consequences: List[str]
    mitigation_actions: List[str]
    calculation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledRepair:
    """Scheduled repair with timing and resources."""
    defect_id: str
    priority_category: PriorityCategory
    scheduled_start: Optional[date]
    scheduled_end: Optional[date]
    outage_required: OutageType
    labor_hours: Decimal
    material_cost: Decimal
    labor_cost: Decimal
    total_cost: Decimal
    batch_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class WorkScope:
    """Detailed work scope for a repair."""
    defect_id: str
    repair_scope: RepairScope
    work_description: str
    material_requirements: List[Dict[str, Any]]
    labor_requirements: List[Dict[str, Any]]
    equipment_requirements: List[str]
    safety_requirements: List[str]
    permits_required: List[str]
    estimated_duration_hours: Decimal
    special_instructions: List[str]
    quality_checkpoints: List[str]


@dataclass
class OptimizedRepairPlan:
    """Complete optimized repair plan."""
    plan_id: str
    generated_at: datetime
    total_defects: int
    total_estimated_cost: Decimal
    total_annual_savings: Decimal
    aggregate_npv: Decimal
    emergency_repairs: List[ScheduledRepair]
    urgent_repairs: List[ScheduledRepair]
    scheduled_repairs: List[ScheduledRepair]
    deferred_repairs: List[ScheduledRepair]
    budget_utilization_percent: Decimal
    work_scopes: List[WorkScope]
    schedule_summary: Dict[str, Any]
    provenance_hash: str


# =============================================================================
# Calculation Step Tracking
# =============================================================================

@dataclass
class CalculationStep:
    """Individual calculation step for audit trail."""
    step_number: int
    description: str
    operation: str
    inputs: Dict[str, Any]
    output_value: Any
    output_name: str
    formula_reference: str = ""


# =============================================================================
# Repair Prioritization Engine
# =============================================================================

class RepairPrioritizationEngine:
    """
    Intelligent repair prioritization engine for insulation maintenance.

    Guarantees:
    - Deterministic: Same input produces same output (bit-perfect)
    - Reproducible: Full provenance tracking
    - Auditable: SHA-256 hash of all calculation steps
    - NO LLM: Zero hallucination risk in calculations

    Methodologies:
    - Multi-factor criticality scoring (weighted composite)
    - ROI-based ranking with NPV analysis
    - FMEA risk prioritization (Severity x Occurrence x Detection)
    - Knapsack optimization for budget constraints
    - Batch optimization for schedule efficiency
    """

    def __init__(
        self,
        criticality_weights: Optional[CriticalityWeights] = None,
        economic_params: Optional[EconomicParameters] = None,
        schedule_constraints: Optional[ScheduleConstraints] = None,
    ) -> None:
        """
        Initialize the repair prioritization engine.

        Args:
            criticality_weights: Custom weights for criticality factors
            economic_params: Economic parameters for ROI calculations
            schedule_constraints: Constraints for schedule optimization
        """
        self.criticality_weights = criticality_weights or CriticalityWeights()
        self.economic_params = economic_params or EconomicParameters()
        self.schedule_constraints = schedule_constraints or ScheduleConstraints()
        self._calculation_steps: List[CalculationStep] = []

    def _reset_calculation_steps(self) -> None:
        """Reset calculation step tracking."""
        self._calculation_steps = []

    def _add_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, Any],
        output_value: Any,
        output_name: str,
        formula_ref: str = ""
    ) -> None:
        """Add a calculation step to the audit trail."""
        step = CalculationStep(
            step_number=len(self._calculation_steps) + 1,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula_reference=formula_ref
        )
        self._calculation_steps.append(step)

    def _apply_precision(self, value: Decimal, precision: int = 3) -> Decimal:
        """Apply regulatory rounding precision."""
        quantize_string = '0.' + '0' * precision
        return value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)

    # =========================================================================
    # 1. Criticality Scoring (Multi-Factor)
    # =========================================================================

    def calculate_criticality_score(self, defect: ThermalDefect) -> CriticalityScore:
        """
        Calculate multi-factor criticality score for a thermal defect.

        Factors:
        - Heat loss severity (0-100): Based on delta-T and W/m
        - Safety risk (0-100): Personnel protection, fire hazard
        - Process impact (0-100): Temperature control, efficiency
        - Environmental (0-100): Emissions, condensation
        - Asset protection (0-100): Corrosion under insulation (CUI)

        Args:
            defect: Thermal defect data

        Returns:
            CriticalityScore with all factor scores and composite
        """
        self._reset_calculation_steps()

        # Calculate individual factor scores
        heat_loss_score = self._calculate_heat_loss_score(defect)
        safety_score = self._calculate_safety_risk_score(defect)
        process_score = self._calculate_process_impact_score(defect)
        environmental_score = self._calculate_environmental_score(defect)
        asset_score = self._calculate_asset_protection_score(defect)

        # Calculate weighted composite score
        composite = (
            heat_loss_score * self.criticality_weights.heat_loss_weight +
            safety_score * self.criticality_weights.safety_risk_weight +
            process_score * self.criticality_weights.process_impact_weight +
            environmental_score * self.criticality_weights.environmental_weight +
            asset_score * self.criticality_weights.asset_protection_weight
        )

        composite = self._apply_precision(composite, 2)

        self._add_step(
            description="Calculate weighted composite criticality score",
            operation="weighted_sum",
            inputs={
                'heat_loss_score': float(heat_loss_score),
                'safety_score': float(safety_score),
                'process_score': float(process_score),
                'environmental_score': float(environmental_score),
                'asset_score': float(asset_score),
                'weights': asdict(self.criticality_weights)
            },
            output_value=float(composite),
            output_name="composite_criticality_score",
            formula_ref="composite = sum(score_i * weight_i)"
        )

        # Determine dominant factor
        scores = {
            'heat_loss': heat_loss_score,
            'safety_risk': safety_score,
            'process_impact': process_score,
            'environmental': environmental_score,
            'asset_protection': asset_score
        }
        dominant_factor = max(scores, key=scores.get)

        return CriticalityScore(
            defect_id=defect.defect_id,
            heat_loss_score=heat_loss_score,
            safety_risk_score=safety_score,
            process_impact_score=process_score,
            environmental_score=environmental_score,
            asset_protection_score=asset_score,
            composite_score=composite,
            dominant_factor=dominant_factor,
            calculation_details={
                'steps': [asdict(s) for s in self._calculation_steps],
                'weights_used': asdict(self.criticality_weights)
            }
        )

    def _calculate_heat_loss_score(self, defect: ThermalDefect) -> Decimal:
        """Calculate heat loss severity score (0-100)."""
        # Determine appropriate threshold set
        is_cold = defect.process_temperature_c < defect.ambient_temperature_c
        equipment_key = 'cold_pipe_damaged' if is_cold else 'hot_pipe_damaged'

        if defect.damage_type == DamageType.MISSING:
            equipment_key = 'cold_pipe_uninsulated' if is_cold else 'hot_pipe_uninsulated'

        thresholds = HEAT_LOSS_SEVERITY_THRESHOLDS.get(
            equipment_key,
            (25.0, 75.0, 150.0, 250.0)
        )

        heat_loss = float(defect.heat_loss_w_per_m)

        # Linear interpolation within threshold bands
        if heat_loss <= thresholds[0]:
            score = Decimal(str((heat_loss / thresholds[0]) * 25))
        elif heat_loss <= thresholds[1]:
            score = Decimal("25") + Decimal(str(
                ((heat_loss - thresholds[0]) / (thresholds[1] - thresholds[0])) * 25
            ))
        elif heat_loss <= thresholds[2]:
            score = Decimal("50") + Decimal(str(
                ((heat_loss - thresholds[1]) / (thresholds[2] - thresholds[1])) * 25
            ))
        elif heat_loss <= thresholds[3]:
            score = Decimal("75") + Decimal(str(
                ((heat_loss - thresholds[2]) / (thresholds[3] - thresholds[2])) * 25
            ))
        else:
            score = Decimal("100")

        score = min(Decimal("100"), max(Decimal("0"), self._apply_precision(score, 2)))

        self._add_step(
            description="Calculate heat loss severity score",
            operation="interpolation",
            inputs={
                'heat_loss_w_per_m': heat_loss,
                'thresholds': thresholds,
                'equipment_key': equipment_key
            },
            output_value=float(score),
            output_name="heat_loss_score",
            formula_ref="Linear interpolation within threshold bands"
        )

        return score

    def _calculate_safety_risk_score(self, defect: ThermalDefect) -> Decimal:
        """Calculate safety risk score (0-100)."""
        score = Decimal("0")

        surface_temp = float(defect.surface_temperature_c)
        process_temp = float(defect.process_temperature_c)

        # Personnel burn risk (hot surfaces per ASTM C1055)
        if surface_temp >= PERSONNEL_SAFETY_TEMP_C['continuous_contact']:
            if surface_temp >= 100.0:
                score += Decimal("50")  # Severe burn risk
            elif surface_temp >= PERSONNEL_SAFETY_TEMP_C['momentary_contact']:
                score += Decimal("40")  # High burn risk
            else:
                score += Decimal("25")  # Moderate burn risk

        # Cold burn risk (cryogenic)
        if surface_temp <= PERSONNEL_SAFETY_TEMP_C['cold_burn_threshold']:
            score += Decimal("50")
        elif surface_temp < 0:
            score += Decimal("30")

        # Fire hazard (high process temps near combustibles)
        if process_temp >= 300.0:
            score += Decimal("30")
        elif process_temp >= 200.0:
            score += Decimal("15")

        # Access difficulty increases risk
        access_multiplier = Decimal(str(1 + (defect.location.access_difficulty - 1) * 0.05))
        score = score * access_multiplier

        # Elevation risk
        if defect.location.elevation_m > 10.0:
            score += Decimal("10")
        elif defect.location.elevation_m > 5.0:
            score += Decimal("5")

        score = min(Decimal("100"), max(Decimal("0"), self._apply_precision(score, 2)))

        self._add_step(
            description="Calculate safety risk score",
            operation="additive_scoring",
            inputs={
                'surface_temp_c': surface_temp,
                'process_temp_c': process_temp,
                'access_difficulty': defect.location.access_difficulty,
                'elevation_m': defect.location.elevation_m
            },
            output_value=float(score),
            output_name="safety_risk_score",
            formula_ref="ASTM C1055 surface temperature limits"
        )

        return score

    def _calculate_process_impact_score(self, defect: ThermalDefect) -> Decimal:
        """Calculate process impact score (0-100)."""
        score = Decimal("0")

        # Temperature deviation impact
        delta_t = abs(float(defect.process_temperature_c) - float(defect.surface_temperature_c))
        expected_delta_t = float(defect.process_temperature_c) - float(defect.ambient_temperature_c)

        if expected_delta_t != 0:
            temp_efficiency_loss = (expected_delta_t - delta_t) / expected_delta_t
            if temp_efficiency_loss > 0.5:
                score += Decimal("40")
            elif temp_efficiency_loss > 0.3:
                score += Decimal("25")
            elif temp_efficiency_loss > 0.15:
                score += Decimal("15")

        # Heat loss magnitude impact
        heat_loss = float(defect.heat_loss_w_per_m)
        length = float(defect.length_m)
        total_loss_kw = (heat_loss * length) / 1000.0

        if total_loss_kw > 50.0:
            score += Decimal("40")
        elif total_loss_kw > 20.0:
            score += Decimal("25")
        elif total_loss_kw > 5.0:
            score += Decimal("15")
        elif total_loss_kw > 1.0:
            score += Decimal("5")

        # Process isolation required increases impact
        if defect.location.process_isolation_required:
            score += Decimal("15")

        # Equipment type criticality
        critical_equipment = {
            EquipmentType.HEAT_EXCHANGER: 10,
            EquipmentType.VESSEL: 8,
            EquipmentType.VALVE: 5,
        }
        score += Decimal(str(critical_equipment.get(defect.location.equipment_type, 0)))

        score = min(Decimal("100"), max(Decimal("0"), self._apply_precision(score, 2)))

        self._add_step(
            description="Calculate process impact score",
            operation="additive_scoring",
            inputs={
                'delta_t': delta_t,
                'total_loss_kw': total_loss_kw,
                'equipment_type': defect.location.equipment_type.value,
                'process_isolation_required': defect.location.process_isolation_required
            },
            output_value=float(score),
            output_name="process_impact_score"
        )

        return score

    def _calculate_environmental_score(self, defect: ThermalDefect) -> Decimal:
        """Calculate environmental impact score (0-100)."""
        score = Decimal("0")

        # CO2 emissions from heat loss
        heat_loss_kw = (float(defect.heat_loss_w_per_m) * float(defect.length_m)) / 1000.0
        annual_kwh = heat_loss_kw * float(self.economic_params.operating_hours_per_year)
        annual_co2_tonnes = (
            annual_kwh * float(self.economic_params.co2_emission_factor_kg_per_kwh) / 1000.0
        )

        if annual_co2_tonnes > 100.0:
            score += Decimal("50")
        elif annual_co2_tonnes > 50.0:
            score += Decimal("35")
        elif annual_co2_tonnes > 20.0:
            score += Decimal("20")
        elif annual_co2_tonnes > 5.0:
            score += Decimal("10")

        # Condensation risk (cold systems)
        if float(defect.surface_temperature_c) < float(defect.ambient_temperature_c):
            dew_point_risk = float(defect.ambient_temperature_c) - float(defect.surface_temperature_c)
            if dew_point_risk > 15.0:
                score += Decimal("30")  # High condensation risk
            elif dew_point_risk > 8.0:
                score += Decimal("20")
            elif dew_point_risk > 3.0:
                score += Decimal("10")

        # Moisture damage type adds risk
        if defect.damage_type in {DamageType.WET, DamageType.VAPOR_BARRIER_FAILED}:
            score += Decimal("20")

        score = min(Decimal("100"), max(Decimal("0"), self._apply_precision(score, 2)))

        self._add_step(
            description="Calculate environmental score",
            operation="additive_scoring",
            inputs={
                'annual_co2_tonnes': annual_co2_tonnes,
                'surface_temp_c': float(defect.surface_temperature_c),
                'ambient_temp_c': float(defect.ambient_temperature_c),
                'damage_type': defect.damage_type.value
            },
            output_value=float(score),
            output_name="environmental_score"
        )

        return score

    def _calculate_asset_protection_score(self, defect: ThermalDefect) -> Decimal:
        """Calculate asset protection score (0-100) focusing on CUI risk."""
        score = Decimal("0")

        process_temp = float(defect.process_temperature_c)

        # Temperature range CUI risk
        if 50.0 <= process_temp <= 150.0:
            # Maximum CUI risk zone (wet-dry cycling)
            score += Decimal("40")
        elif 150.0 < process_temp <= 250.0:
            score += Decimal("25")
        elif 0.0 <= process_temp < 50.0:
            score += Decimal("15")
        elif process_temp < 0.0:
            score += Decimal("20")  # Ice formation risk
        else:
            score += Decimal("10")  # Above 250C, dry conditions

        # Insulation condition factor
        condition_score = defect.existing_condition_score
        score += Decimal(str(condition_score * 3))  # Max 30 points

        # Damage type CUI indicators
        cui_damage_types = {
            DamageType.WET: 25,
            DamageType.VAPOR_BARRIER_FAILED: 20,
            DamageType.JACKET_DAMAGED: 15,
            DamageType.CUI_SUSPECTED: 30,
        }
        score += Decimal(str(cui_damage_types.get(defect.damage_type, 0)))

        # Equipment material susceptibility
        if defect.location.equipment_type in {
            EquipmentType.PIPE,
            EquipmentType.VESSEL,
            EquipmentType.TANK
        }:
            score += Decimal("5")  # Carbon steel susceptible

        score = min(Decimal("100"), max(Decimal("0"), self._apply_precision(score, 2)))

        self._add_step(
            description="Calculate asset protection (CUI) score",
            operation="additive_scoring",
            inputs={
                'process_temp_c': process_temp,
                'condition_score': condition_score,
                'damage_type': defect.damage_type.value,
                'equipment_type': defect.location.equipment_type.value
            },
            output_value=float(score),
            output_name="asset_protection_score",
            formula_ref="NACE SP0198 CUI risk factors"
        )

        return score

    # =========================================================================
    # 2. ROI-Based Ranking
    # =========================================================================

    def calculate_repair_roi(self, defect: ThermalDefect) -> RepairROI:
        """
        Calculate ROI metrics for a repair.

        Includes:
        - Repair cost estimation by insulation type/length
        - Annual energy savings calculation
        - Simple payback period
        - NPV over equipment life
        - ROI percentage

        Args:
            defect: Thermal defect data

        Returns:
            RepairROI with all financial metrics
        """
        self._reset_calculation_steps()

        # Step 1: Estimate repair cost
        repair_cost = self._estimate_repair_cost(defect)

        # Step 2: Calculate annual energy savings
        annual_savings = self._calculate_annual_energy_savings(defect)

        # Step 3: Calculate carbon savings
        carbon_savings = self._calculate_carbon_savings(defect)

        # Step 4: Simple payback
        if annual_savings > Decimal("0"):
            simple_payback = repair_cost / annual_savings
        else:
            simple_payback = Decimal("999")  # Infinite payback

        self._add_step(
            description="Calculate simple payback period",
            operation="division",
            inputs={
                'repair_cost': float(repair_cost),
                'annual_savings': float(annual_savings)
            },
            output_value=float(simple_payback),
            output_name="simple_payback_years",
            formula_ref="Payback = Cost / Annual_Savings"
        )

        # Step 5: NPV calculation
        npv = self._calculate_npv(repair_cost, annual_savings)

        # Step 6: IRR calculation (approximation)
        irr = self._calculate_irr_approximation(repair_cost, annual_savings)

        # Step 7: ROI percentage
        if repair_cost > Decimal("0"):
            total_savings = annual_savings * Decimal(str(self.economic_params.equipment_life_years))
            roi_percent = ((total_savings - repair_cost) / repair_cost) * Decimal("100")
        else:
            roi_percent = Decimal("0")

        roi_percent = self._apply_precision(roi_percent, 2)

        self._add_step(
            description="Calculate ROI percentage",
            operation="roi_calculation",
            inputs={
                'total_savings': float(total_savings) if repair_cost > 0 else 0,
                'repair_cost': float(repair_cost)
            },
            output_value=float(roi_percent),
            output_name="roi_percent",
            formula_ref="ROI = ((Total_Savings - Cost) / Cost) * 100"
        )

        # Cost per unit saved
        if annual_savings > Decimal("0"):
            life_years = Decimal(str(self.economic_params.equipment_life_years))
            total_kwh_saved = (
                self._calculate_heat_loss_kw(defect) *
                self.economic_params.operating_hours_per_year *
                life_years
            )
            if total_kwh_saved > Decimal("0"):
                cost_per_kwh = repair_cost / total_kwh_saved
            else:
                cost_per_kwh = Decimal("999")
        else:
            cost_per_kwh = Decimal("999")

        return RepairROI(
            defect_id=defect.defect_id,
            estimated_repair_cost=repair_cost,
            annual_energy_savings=annual_savings,
            annual_carbon_savings_tonnes=carbon_savings,
            simple_payback_years=self._apply_precision(simple_payback, 2),
            npv_over_life=npv,
            irr_percent=irr,
            roi_percent=roi_percent,
            cost_per_unit_saved=self._apply_precision(cost_per_kwh, 4),
            calculation_details={
                'steps': [asdict(s) for s in self._calculation_steps],
                'economic_params': asdict(self.economic_params)
            }
        )

    def _estimate_repair_cost(self, defect: ThermalDefect) -> Decimal:
        """Estimate repair cost based on material, length, and complexity."""
        # Get material cost per meter
        material = defect.insulation_material.value
        thickness = int(defect.insulation_thickness_mm)

        cost_table = INSULATION_COST_PER_METER.get(material, {})

        # Find closest thickness
        available_thicknesses = sorted(cost_table.keys())
        if not available_thicknesses:
            base_cost_per_m = Decimal("30.0")  # Default
        else:
            closest_thickness = min(
                available_thicknesses,
                key=lambda x: abs(x - thickness)
            )
            base_cost_per_m = Decimal(str(cost_table[closest_thickness]))

        # Material cost
        length = defect.length_m
        material_cost = base_cost_per_m * length

        # Apply material markup
        material_cost *= (Decimal("1") + self.economic_params.material_markup_percent / Decimal("100"))

        # Estimate labor hours
        scope = self._determine_repair_scope(defect)
        if scope == RepairScope.SPOT_REPAIR:
            production_rate = Decimal(str(PRODUCTION_RATES['insulation_installation_complex']))
        elif scope == RepairScope.SECTION_REPAIR:
            production_rate = Decimal(str(PRODUCTION_RATES['insulation_installation_simple']))
        else:
            production_rate = Decimal(str(PRODUCTION_RATES['insulation_installation_simple']))

        labor_hours = length / production_rate

        # Add removal time if replacing
        if defect.damage_type != DamageType.MISSING:
            removal_rate = Decimal(str(PRODUCTION_RATES['removal_damaged']))
            labor_hours += length / removal_rate

        # Add jacketing time
        jacket_rate = Decimal(str(PRODUCTION_RATES['jacketing_aluminum']))
        labor_hours += length / jacket_rate

        # Labor cost
        labor_rate = Decimal(str(LABOR_RATES['journeyman_insulator']))
        labor_cost = labor_hours * labor_rate

        # Apply overhead
        labor_cost *= (Decimal("1") + self.economic_params.labor_overhead_percent / Decimal("100"))

        # Access difficulty factor
        access_factor = Decimal(str(1 + (defect.location.access_difficulty - 1) * 0.15))
        labor_cost *= access_factor

        # Scaffolding cost
        scaffold_cost = Decimal("0")
        if defect.location.scaffold_required:
            scaffold_cost = length * Decimal("50")  # $50/m for scaffold

        total_cost = material_cost + labor_cost + scaffold_cost
        total_cost = self._apply_precision(total_cost, 2)

        self._add_step(
            description="Estimate repair cost",
            operation="cost_estimation",
            inputs={
                'material': material,
                'length_m': float(length),
                'base_cost_per_m': float(base_cost_per_m),
                'labor_hours': float(labor_hours),
                'scaffold_required': defect.location.scaffold_required
            },
            output_value=float(total_cost),
            output_name="estimated_repair_cost"
        )

        return total_cost

    def _calculate_heat_loss_kw(self, defect: ThermalDefect) -> Decimal:
        """Calculate total heat loss in kW."""
        heat_loss_w = defect.heat_loss_w_per_m * defect.length_m
        return heat_loss_w / Decimal("1000")

    def _calculate_annual_energy_savings(self, defect: ThermalDefect) -> Decimal:
        """Calculate annual energy cost savings from repair."""
        heat_loss_kw = self._calculate_heat_loss_kw(defect)

        # Assume repair restores 90% of design performance
        restoration_factor = Decimal("0.90")
        saved_kw = heat_loss_kw * restoration_factor

        # Annual energy saved
        annual_kwh = saved_kw * self.economic_params.operating_hours_per_year

        # Energy cost savings
        energy_savings = annual_kwh * self.economic_params.energy_cost_per_kwh

        # Add carbon cost savings
        carbon_savings_tonnes = (
            annual_kwh * self.economic_params.co2_emission_factor_kg_per_kwh / Decimal("1000")
        )
        carbon_cost_savings = carbon_savings_tonnes * self.economic_params.carbon_price_per_tonne

        total_savings = energy_savings + carbon_cost_savings
        total_savings = self._apply_precision(total_savings, 2)

        self._add_step(
            description="Calculate annual energy savings",
            operation="energy_calculation",
            inputs={
                'heat_loss_kw': float(heat_loss_kw),
                'restoration_factor': float(restoration_factor),
                'operating_hours': float(self.economic_params.operating_hours_per_year),
                'energy_cost': float(self.economic_params.energy_cost_per_kwh),
                'carbon_price': float(self.economic_params.carbon_price_per_tonne)
            },
            output_value=float(total_savings),
            output_name="annual_energy_savings"
        )

        return total_savings

    def _calculate_carbon_savings(self, defect: ThermalDefect) -> Decimal:
        """Calculate annual CO2 savings in tonnes."""
        heat_loss_kw = self._calculate_heat_loss_kw(defect)
        restoration_factor = Decimal("0.90")
        saved_kw = heat_loss_kw * restoration_factor

        annual_kwh = saved_kw * self.economic_params.operating_hours_per_year
        carbon_savings = (
            annual_kwh * self.economic_params.co2_emission_factor_kg_per_kwh / Decimal("1000")
        )

        return self._apply_precision(carbon_savings, 3)

    def _calculate_npv(self, initial_cost: Decimal, annual_savings: Decimal) -> Decimal:
        """Calculate Net Present Value over equipment life."""
        discount_rate = self.economic_params.discount_rate_percent / Decimal("100")
        years = self.economic_params.equipment_life_years

        npv = -initial_cost

        for year in range(1, years + 1):
            discount_factor = (Decimal("1") + discount_rate) ** year
            npv += annual_savings / discount_factor

        npv = self._apply_precision(npv, 2)

        self._add_step(
            description="Calculate NPV over equipment life",
            operation="npv_calculation",
            inputs={
                'initial_cost': float(initial_cost),
                'annual_savings': float(annual_savings),
                'discount_rate': float(discount_rate),
                'years': years
            },
            output_value=float(npv),
            output_name="npv_over_life",
            formula_ref="NPV = -Cost + Sum(Savings_t / (1+r)^t)"
        )

        return npv

    def _calculate_irr_approximation(
        self,
        initial_cost: Decimal,
        annual_savings: Decimal
    ) -> Optional[Decimal]:
        """Approximate IRR using Newton-Raphson method."""
        if initial_cost <= Decimal("0") or annual_savings <= Decimal("0"):
            return None

        years = self.economic_params.equipment_life_years

        # Initial guess based on simple payback
        if annual_savings > Decimal("0"):
            payback = float(initial_cost / annual_savings)
            if payback < years:
                irr_guess = 1.0 / payback - 0.1
            else:
                return Decimal("0")
        else:
            return None

        irr_guess = max(0.01, min(irr_guess, 1.0))

        # Newton-Raphson iteration
        for _ in range(50):
            npv = -float(initial_cost)
            npv_derivative = 0.0

            for t in range(1, years + 1):
                factor = (1 + irr_guess) ** t
                npv += float(annual_savings) / factor
                npv_derivative -= t * float(annual_savings) / ((1 + irr_guess) ** (t + 1))

            if abs(npv_derivative) < 1e-10:
                break

            irr_new = irr_guess - npv / npv_derivative

            if abs(irr_new - irr_guess) < 0.0001:
                break

            irr_guess = max(-0.99, min(irr_new, 10.0))

        irr_percent = Decimal(str(irr_guess * 100))
        return self._apply_precision(irr_percent, 2)

    def _determine_repair_scope(self, defect: ThermalDefect) -> RepairScope:
        """Determine repair scope based on defect size."""
        length = float(defect.length_m)

        if length < 1.0:
            return RepairScope.SPOT_REPAIR
        elif length < 10.0:
            return RepairScope.SECTION_REPAIR
        elif length < 50.0:
            return RepairScope.ZONE_REPLACEMENT
        else:
            return RepairScope.FULL_REPLACEMENT

    # =========================================================================
    # 3. Risk-Based Prioritization
    # =========================================================================

    def assess_risk_priority(self, defect: ThermalDefect) -> RiskAssessment:
        """
        Assess risk using FMEA methodology.

        Risk Priority Number (RPN) = Severity x Occurrence x Detection

        Args:
            defect: Thermal defect data

        Returns:
            RiskAssessment with RPN and risk classification
        """
        self._reset_calculation_steps()

        # Determine ratings
        severity = self._assess_severity(defect)
        occurrence = self._assess_occurrence(defect)
        detection = self._assess_detection(defect)

        # Calculate RPN
        rpn = severity.value * occurrence.value * detection.value

        self._add_step(
            description="Calculate Risk Priority Number",
            operation="multiplication",
            inputs={
                'severity': severity.value,
                'occurrence': occurrence.value,
                'detection': detection.value
            },
            output_value=rpn,
            output_name="risk_priority_number",
            formula_ref="RPN = S x O x D (MIL-STD-1629A)"
        )

        # Classify risk level
        risk_level = self._classify_risk_level(rpn, severity)

        # Identify failure modes
        failure_modes = self._identify_failure_modes(defect)

        # Identify consequences
        consequences = self._identify_consequences(defect)

        # Recommend mitigation actions
        mitigation_actions = self._recommend_mitigations(defect, risk_level)

        return RiskAssessment(
            defect_id=defect.defect_id,
            severity_rating=severity,
            occurrence_rating=occurrence,
            detection_rating=detection,
            risk_priority_number=rpn,
            risk_level=risk_level,
            failure_modes=failure_modes,
            consequences=consequences,
            mitigation_actions=mitigation_actions,
            calculation_details={
                'steps': [asdict(s) for s in self._calculation_steps]
            }
        )

    def _assess_severity(self, defect: ThermalDefect) -> SeverityRating:
        """Assess severity rating based on consequences."""
        surface_temp = float(defect.surface_temperature_c)
        process_temp = float(defect.process_temperature_c)

        # Safety-critical severities
        if surface_temp >= 100.0 or surface_temp <= -30.0:
            return SeverityRating.HAZARDOUS

        if surface_temp >= PERSONNEL_SAFETY_TEMP_C['momentary_contact']:
            return SeverityRating.CRITICAL

        # CUI risk severity
        if defect.damage_type == DamageType.CUI_SUSPECTED:
            return SeverityRating.VERY_HIGH

        # High process impact
        if defect.location.process_isolation_required:
            return SeverityRating.HIGH

        # Heat loss severity
        heat_loss = float(defect.heat_loss_w_per_m)
        if heat_loss > 300.0:
            return SeverityRating.MODERATE_HIGH
        elif heat_loss > 150.0:
            return SeverityRating.MODERATE
        elif heat_loss > 75.0:
            return SeverityRating.MODERATE_LOW
        elif heat_loss > 25.0:
            return SeverityRating.MINOR
        else:
            return SeverityRating.VERY_MINOR

    def _assess_occurrence(self, defect: ThermalDefect) -> OccurrenceRating:
        """Assess occurrence/probability rating."""
        condition = defect.existing_condition_score

        # Map condition score to occurrence rating
        if condition >= 9:
            return OccurrenceRating.INEVITABLE
        elif condition >= 8:
            return OccurrenceRating.ALMOST_CERTAIN
        elif condition >= 7:
            return OccurrenceRating.VERY_HIGH
        elif condition >= 6:
            return OccurrenceRating.HIGH
        elif condition >= 5:
            return OccurrenceRating.MODERATE_HIGH
        elif condition >= 4:
            return OccurrenceRating.MODERATE
        elif condition >= 3:
            return OccurrenceRating.MODERATE_LOW
        elif condition >= 2:
            return OccurrenceRating.LOW
        else:
            return OccurrenceRating.VERY_LOW

    def _assess_detection(self, defect: ThermalDefect) -> DetectionRating:
        """Assess detection difficulty rating."""
        # Thermal imaging provides good detection
        base_detection = DetectionRating.MODERATELY_HIGH

        # Adjust for access difficulty
        if defect.location.access_difficulty >= 4:
            return DetectionRating.LOW
        elif defect.location.access_difficulty >= 3:
            return DetectionRating.MODERATELY_LOW

        # CUI is harder to detect
        if defect.damage_type == DamageType.CUI_SUSPECTED:
            return DetectionRating.VERY_LOW

        # Vapor barrier failures are subtle
        if defect.damage_type == DamageType.VAPOR_BARRIER_FAILED:
            return DetectionRating.LOW

        return base_detection

    def _classify_risk_level(self, rpn: int, severity: SeverityRating) -> RiskLevel:
        """Classify risk level based on RPN and severity."""
        # High severity overrides RPN for critical items
        if severity == SeverityRating.HAZARDOUS:
            return RiskLevel.EXTREME
        if severity == SeverityRating.CRITICAL:
            return RiskLevel.HIGH

        # RPN-based classification
        if rpn >= 200:
            return RiskLevel.EXTREME
        elif rpn >= 120:
            return RiskLevel.HIGH
        elif rpn >= 60:
            return RiskLevel.MEDIUM
        elif rpn >= 20:
            return RiskLevel.LOW
        else:
            return RiskLevel.NEGLIGIBLE

    def _identify_failure_modes(self, defect: ThermalDefect) -> List[str]:
        """Identify potential failure modes."""
        failure_modes = []

        damage_failure_modes = {
            DamageType.MISSING: ["Uncontrolled heat loss", "Personnel burn hazard"],
            DamageType.COMPRESSED: ["Reduced thermal resistance", "Vapor barrier breach"],
            DamageType.WET: ["Corrosion initiation", "Ice formation", "Mold growth"],
            DamageType.CRACKED: ["Moisture ingress", "Progressive degradation"],
            DamageType.JACKET_DAMAGED: ["Weather exposure", "Moisture ingress"],
            DamageType.VAPOR_BARRIER_FAILED: ["Condensation", "Ice formation"],
            DamageType.THERMAL_BRIDGING: ["Localized heat loss", "Condensation points"],
            DamageType.CUI_SUSPECTED: ["Wall thinning", "Leak potential", "Structural failure"],
        }

        failure_modes.extend(damage_failure_modes.get(defect.damage_type, []))

        # Temperature-specific failure modes
        if float(defect.process_temperature_c) > 200.0:
            failure_modes.append("Fire hazard from heat transfer")

        if float(defect.surface_temperature_c) >= PERSONNEL_SAFETY_TEMP_C['momentary_contact']:
            failure_modes.append("Personnel thermal injury")

        return failure_modes

    def _identify_consequences(self, defect: ThermalDefect) -> List[str]:
        """Identify consequences of failure."""
        consequences = []

        # Energy consequences
        heat_loss_kw = float(self._calculate_heat_loss_kw(defect))
        annual_cost = (
            heat_loss_kw *
            float(self.economic_params.operating_hours_per_year) *
            float(self.economic_params.energy_cost_per_kwh)
        )
        consequences.append(f"Energy loss: ${annual_cost:,.0f}/year")

        # Safety consequences
        if float(defect.surface_temperature_c) >= PERSONNEL_SAFETY_TEMP_C['momentary_contact']:
            consequences.append("Potential burn injury")

        # Asset consequences
        if defect.damage_type in {DamageType.WET, DamageType.CUI_SUSPECTED}:
            consequences.append("Corrosion damage to equipment")

        # Environmental consequences
        carbon_tonnes = float(self._calculate_carbon_savings(defect))
        if carbon_tonnes > 10.0:
            consequences.append(f"Excess CO2 emissions: {carbon_tonnes:.1f} t/year")

        return consequences

    def _recommend_mitigations(self, defect: ThermalDefect, risk_level: RiskLevel) -> List[str]:
        """Recommend mitigation actions based on risk level."""
        mitigations = []

        if risk_level == RiskLevel.EXTREME:
            mitigations.append("IMMEDIATE: Isolate area and implement emergency repair")
            mitigations.append("Install temporary protective barriers")
        elif risk_level == RiskLevel.HIGH:
            mitigations.append("Schedule repair within 30 days")
            mitigations.append("Increase inspection frequency")
        elif risk_level == RiskLevel.MEDIUM:
            mitigations.append("Schedule repair at next turnaround")
            mitigations.append("Monitor condition monthly")
        elif risk_level == RiskLevel.LOW:
            mitigations.append("Include in routine maintenance program")
            mitigations.append("Reinspect in 12 months")
        else:
            mitigations.append("No action required")
            mitigations.append("Continue normal inspection schedule")

        # Specific mitigations by damage type
        if defect.damage_type == DamageType.CUI_SUSPECTED:
            mitigations.append("Perform invasive inspection to confirm CUI")
            mitigations.append("Consider pipe thickness measurement")

        if defect.damage_type == DamageType.WET:
            mitigations.append("Remove wet insulation to prevent further corrosion")
            mitigations.append("Dry and inspect underlying surface")

        return mitigations

    # =========================================================================
    # 4. Scheduling Optimization
    # =========================================================================

    def optimize_repair_schedule(
        self,
        defects: List[ThermalDefect],
        criticality_scores: List[CriticalityScore],
        roi_results: List[RepairROI],
        risk_assessments: List[RiskAssessment]
    ) -> List[ScheduledRepair]:
        """
        Optimize repair schedule considering:
        - Group repairs by location (reduce mobilization)
        - Group by insulation type (material efficiency)
        - Align with planned outages/turnarounds
        - Resource leveling (labor, materials)
        - Critical path identification

        Args:
            defects: List of thermal defects
            criticality_scores: Criticality scores for each defect
            roi_results: ROI calculations for each defect
            risk_assessments: Risk assessments for each defect

        Returns:
            List of scheduled repairs with timing and resources
        """
        self._reset_calculation_steps()

        scheduled_repairs: List[ScheduledRepair] = []

        # Create lookup maps
        criticality_map = {cs.defect_id: cs for cs in criticality_scores}
        roi_map = {roi.defect_id: roi for roi in roi_results}
        risk_map = {ra.defect_id: ra for ra in risk_assessments}

        # Step 1: Assign priority categories
        categorized_defects: Dict[PriorityCategory, List[ThermalDefect]] = defaultdict(list)

        for defect in defects:
            category = self.assign_priority_category(
                defect,
                criticality_map.get(defect.defect_id),
                risk_map.get(defect.defect_id)
            )
            categorized_defects[category].append(defect)

        # Step 2: Group by location for batch efficiency
        location_groups = self._group_by_location(defects)

        self._add_step(
            description="Group defects by location",
            operation="grouping",
            inputs={'total_defects': len(defects)},
            output_value=len(location_groups),
            output_name="location_groups"
        )

        # Step 3: Schedule each category
        current_date = date.today()

        # Emergency repairs - immediate
        for defect in categorized_defects[PriorityCategory.EMERGENCY]:
            repair = self._create_scheduled_repair(
                defect,
                PriorityCategory.EMERGENCY,
                current_date,
                roi_map.get(defect.defect_id)
            )
            scheduled_repairs.append(repair)

        # Urgent repairs - within 30 days
        urgent_start = current_date + timedelta(days=1)
        for defect in categorized_defects[PriorityCategory.URGENT]:
            repair = self._create_scheduled_repair(
                defect,
                PriorityCategory.URGENT,
                urgent_start,
                roi_map.get(defect.defect_id)
            )
            scheduled_repairs.append(repair)
            urgent_start += timedelta(days=2)  # Stagger

        # High priority - align with turnaround if available
        if self.schedule_constraints.next_turnaround_date:
            turnaround_date = self.schedule_constraints.next_turnaround_date
        else:
            turnaround_date = current_date + timedelta(days=90)

        for defect in categorized_defects[PriorityCategory.HIGH]:
            repair = self._create_scheduled_repair(
                defect,
                PriorityCategory.HIGH,
                turnaround_date,
                roi_map.get(defect.defect_id)
            )
            scheduled_repairs.append(repair)

        # Medium priority - within 12 months
        medium_start = current_date + timedelta(days=180)
        for defect in categorized_defects[PriorityCategory.MEDIUM]:
            repair = self._create_scheduled_repair(
                defect,
                PriorityCategory.MEDIUM,
                medium_start,
                roi_map.get(defect.defect_id)
            )
            scheduled_repairs.append(repair)

        # Low priority - routine maintenance
        low_start = current_date + timedelta(days=365)
        for defect in categorized_defects[PriorityCategory.LOW]:
            repair = self._create_scheduled_repair(
                defect,
                PriorityCategory.LOW,
                low_start,
                roi_map.get(defect.defect_id)
            )
            scheduled_repairs.append(repair)

        # Monitor items - no scheduled repair
        for defect in categorized_defects[PriorityCategory.MONITOR]:
            repair = ScheduledRepair(
                defect_id=defect.defect_id,
                priority_category=PriorityCategory.MONITOR,
                scheduled_start=None,
                scheduled_end=None,
                outage_required=OutageType.RUNNING,
                labor_hours=Decimal("0"),
                material_cost=Decimal("0"),
                labor_cost=Decimal("0"),
                total_cost=Decimal("0")
            )
            scheduled_repairs.append(repair)

        # Step 4: Assign batch IDs for location grouping
        scheduled_repairs = self._assign_batch_ids(scheduled_repairs, defects, location_groups)

        return scheduled_repairs

    def _group_by_location(
        self,
        defects: List[ThermalDefect]
    ) -> Dict[str, List[ThermalDefect]]:
        """Group defects by location for batching."""
        groups: Dict[str, List[ThermalDefect]] = defaultdict(list)

        for defect in defects:
            location_key = f"{defect.location.unit_code}_{defect.location.area_code}"
            groups[location_key].append(defect)

        return dict(groups)

    def _create_scheduled_repair(
        self,
        defect: ThermalDefect,
        category: PriorityCategory,
        start_date: date,
        roi: Optional[RepairROI]
    ) -> ScheduledRepair:
        """Create a scheduled repair entry."""
        # Determine outage requirement
        if defect.location.process_isolation_required:
            outage = OutageType.SHUTDOWN
        elif category == PriorityCategory.EMERGENCY:
            outage = OutageType.EMERGENCY
        elif category == PriorityCategory.HIGH:
            outage = OutageType.TURNAROUND
        else:
            outage = OutageType.RUNNING

        # Estimate labor hours
        length = float(defect.length_m)
        base_hours = length / PRODUCTION_RATES['insulation_installation_simple']

        # Add removal if not missing
        if defect.damage_type != DamageType.MISSING:
            base_hours += length / PRODUCTION_RATES['removal_damaged']

        # Add jacketing
        base_hours += length / PRODUCTION_RATES['jacketing_aluminum']

        # Access difficulty factor
        access_factor = 1 + (defect.location.access_difficulty - 1) * 0.2
        labor_hours = Decimal(str(base_hours * access_factor))

        # Costs from ROI or estimate
        if roi:
            material_cost = roi.estimated_repair_cost * Decimal("0.4")
            labor_cost = roi.estimated_repair_cost * Decimal("0.6")
            total_cost = roi.estimated_repair_cost
        else:
            labor_rate = Decimal(str(LABOR_RATES['journeyman_insulator']))
            labor_cost = labor_hours * labor_rate
            material_cost = defect.length_m * Decimal("30")  # Default
            total_cost = labor_cost + material_cost

        # Calculate end date
        work_days = int(float(labor_hours) / float(self.schedule_constraints.shift_hours)) + 1
        end_date = start_date + timedelta(days=work_days)

        return ScheduledRepair(
            defect_id=defect.defect_id,
            priority_category=category,
            scheduled_start=start_date,
            scheduled_end=end_date,
            outage_required=outage,
            labor_hours=self._apply_precision(labor_hours, 1),
            material_cost=self._apply_precision(material_cost, 2),
            labor_cost=self._apply_precision(labor_cost, 2),
            total_cost=self._apply_precision(total_cost, 2)
        )

    def _assign_batch_ids(
        self,
        repairs: List[ScheduledRepair],
        defects: List[ThermalDefect],
        location_groups: Dict[str, List[ThermalDefect]]
    ) -> List[ScheduledRepair]:
        """Assign batch IDs for grouped repairs."""
        defect_map = {d.defect_id: d for d in defects}

        updated_repairs = []
        batch_counter = 1

        for location_key, group_defects in location_groups.items():
            group_defect_ids = {d.defect_id for d in group_defects}

            if len(group_defects) >= 2:
                batch_id = f"BATCH-{batch_counter:04d}"
                batch_counter += 1
            else:
                batch_id = None

            for repair in repairs:
                if repair.defect_id in group_defect_ids:
                    updated_repair = ScheduledRepair(
                        defect_id=repair.defect_id,
                        priority_category=repair.priority_category,
                        scheduled_start=repair.scheduled_start,
                        scheduled_end=repair.scheduled_end,
                        outage_required=repair.outage_required,
                        labor_hours=repair.labor_hours,
                        material_cost=repair.material_cost,
                        labor_cost=repair.labor_cost,
                        total_cost=repair.total_cost,
                        batch_id=batch_id,
                        dependencies=repair.dependencies
                    )
                    updated_repairs.append(updated_repair)
                    group_defect_ids.discard(repair.defect_id)

        # Add any repairs not in groups
        existing_ids = {r.defect_id for r in updated_repairs}
        for repair in repairs:
            if repair.defect_id not in existing_ids:
                updated_repairs.append(repair)

        return updated_repairs

    # =========================================================================
    # 5. Priority Categories
    # =========================================================================

    def assign_priority_category(
        self,
        defect: ThermalDefect,
        criticality: Optional[CriticalityScore],
        risk: Optional[RiskAssessment]
    ) -> PriorityCategory:
        """
        Assign priority category based on criticality and risk.

        Categories:
        - EMERGENCY: Safety hazard, immediate action
        - URGENT: High heat loss, repair within 30 days
        - HIGH: Significant degradation, next turnaround
        - MEDIUM: Moderate degradation, within 12 months
        - LOW: Minor issues, routine maintenance
        - MONITOR: Acceptable condition, reinspect

        Args:
            defect: Thermal defect data
            criticality: Criticality score (optional)
            risk: Risk assessment (optional)

        Returns:
            PriorityCategory
        """
        # Safety overrides - immediate emergency
        surface_temp = float(defect.surface_temperature_c)
        if surface_temp >= 100.0 or surface_temp <= -30.0:
            return PriorityCategory.EMERGENCY

        # Check risk level
        if risk:
            if risk.risk_level == RiskLevel.EXTREME:
                return PriorityCategory.EMERGENCY
            if risk.risk_level == RiskLevel.HIGH:
                return PriorityCategory.URGENT

        # Check criticality score
        if criticality:
            composite = float(criticality.composite_score)

            if composite >= 85:
                return PriorityCategory.EMERGENCY
            elif composite >= 70:
                return PriorityCategory.URGENT
            elif composite >= 50:
                return PriorityCategory.HIGH
            elif composite >= 30:
                return PriorityCategory.MEDIUM
            elif composite >= 15:
                return PriorityCategory.LOW
            else:
                return PriorityCategory.MONITOR

        # Default based on damage type
        urgent_damage = {DamageType.CUI_SUSPECTED, DamageType.WET}
        if defect.damage_type in urgent_damage:
            return PriorityCategory.URGENT

        if defect.damage_type == DamageType.MISSING:
            return PriorityCategory.HIGH

        # Default based on condition score
        if defect.existing_condition_score >= 8:
            return PriorityCategory.HIGH
        elif defect.existing_condition_score >= 5:
            return PriorityCategory.MEDIUM
        else:
            return PriorityCategory.LOW

    # =========================================================================
    # 6. Budget Optimization
    # =========================================================================

    def optimize_within_budget(
        self,
        repairs: List[ScheduledRepair],
        roi_results: List[RepairROI],
        budget: BudgetConstraint
    ) -> Tuple[List[ScheduledRepair], List[ScheduledRepair]]:
        """
        Optimize repair selection within budget constraint.

        Uses 0/1 knapsack optimization to maximize benefit.

        Args:
            repairs: List of scheduled repairs
            roi_results: ROI calculations for each repair
            budget: Budget constraint parameters

        Returns:
            Tuple of (selected repairs, deferred repairs)
        """
        self._reset_calculation_steps()

        # Calculate available budget
        total_budget = budget.total_budget
        emergency_reserve = total_budget * budget.emergency_reserve_percent / Decimal("100")
        available_budget = total_budget - emergency_reserve

        self._add_step(
            description="Calculate available budget",
            operation="subtraction",
            inputs={
                'total_budget': float(total_budget),
                'emergency_reserve_percent': float(budget.emergency_reserve_percent)
            },
            output_value=float(available_budget),
            output_name="available_budget"
        )

        # Create ROI lookup
        roi_map = {roi.defect_id: roi for roi in roi_results}

        # Separate emergency repairs (must do regardless of budget)
        emergency_repairs = [
            r for r in repairs
            if r.priority_category == PriorityCategory.EMERGENCY
        ]
        emergency_cost = sum(r.total_cost for r in emergency_repairs)

        # Remaining budget after emergencies
        remaining_budget = available_budget - emergency_cost

        # Non-emergency repairs for optimization
        candidate_repairs = [
            r for r in repairs
            if r.priority_category != PriorityCategory.EMERGENCY
            and r.priority_category != PriorityCategory.MONITOR
        ]

        # Calculate benefit scores (NPV-based)
        benefits = []
        for repair in candidate_repairs:
            roi = roi_map.get(repair.defect_id)
            if roi:
                benefit = float(roi.npv_over_life) if roi.npv_over_life > 0 else 0.0
            else:
                benefit = float(repair.total_cost) * 0.5  # Default benefit estimate
            benefits.append(benefit)

        costs = [float(r.total_cost) for r in candidate_repairs]

        # Solve knapsack problem
        selected_indices = self._solve_knapsack(
            benefits,
            costs,
            float(remaining_budget)
        )

        self._add_step(
            description="Solve knapsack optimization",
            operation="dynamic_programming",
            inputs={
                'num_candidates': len(candidate_repairs),
                'remaining_budget': float(remaining_budget)
            },
            output_value=len(selected_indices),
            output_name="selected_repairs_count"
        )

        # Separate selected and deferred
        selected_repairs = emergency_repairs.copy()
        deferred_repairs = []

        for i, repair in enumerate(candidate_repairs):
            if i in selected_indices:
                selected_repairs.append(repair)
            else:
                deferred_repairs.append(repair)

        # Add monitor items to deferred
        monitor_repairs = [
            r for r in repairs
            if r.priority_category == PriorityCategory.MONITOR
        ]
        deferred_repairs.extend(monitor_repairs)

        return selected_repairs, deferred_repairs

    def _solve_knapsack(
        self,
        benefits: List[float],
        costs: List[float],
        capacity: float
    ) -> Set[int]:
        """
        Solve 0/1 knapsack problem using dynamic programming.

        Deterministic algorithm - same inputs always produce same output.
        """
        n = len(benefits)
        if n == 0:
            return set()

        # Scale to integers for DP (use cents)
        scale = 100
        int_costs = [int(c * scale) for c in costs]
        int_capacity = int(capacity * scale)

        # DP table
        dp = [[0.0] * (int_capacity + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(int_capacity + 1):
                if int_costs[i-1] <= w:
                    dp[i][w] = max(
                        dp[i-1][w],
                        dp[i-1][w - int_costs[i-1]] + benefits[i-1]
                    )
                else:
                    dp[i][w] = dp[i-1][w]

        # Backtrack to find selected items
        selected = set()
        w = int_capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.add(i - 1)
                w -= int_costs[i-1]

        return selected

    def identify_quick_wins(
        self,
        repairs: List[ScheduledRepair],
        roi_results: List[RepairROI],
        max_payback_years: float = 2.0,
        max_cost: Decimal = Decimal("10000.0")
    ) -> List[ScheduledRepair]:
        """
        Identify quick wins - high ROI, low cost repairs.

        Args:
            repairs: List of scheduled repairs
            roi_results: ROI calculations
            max_payback_years: Maximum acceptable payback period
            max_cost: Maximum cost threshold

        Returns:
            List of quick win repairs
        """
        roi_map = {roi.defect_id: roi for roi in roi_results}
        quick_wins = []

        for repair in repairs:
            roi = roi_map.get(repair.defect_id)
            if roi:
                if (
                    float(roi.simple_payback_years) <= max_payback_years and
                    repair.total_cost <= max_cost and
                    roi.npv_over_life > Decimal("0")
                ):
                    quick_wins.append(repair)

        # Sort by ROI (descending)
        quick_wins.sort(
            key=lambda r: float(roi_map[r.defect_id].roi_percent),
            reverse=True
        )

        return quick_wins

    # =========================================================================
    # 7. Work Order Generation
    # =========================================================================

    def generate_work_scope(self, defect: ThermalDefect) -> WorkScope:
        """
        Generate detailed work scope for a repair.

        Includes:
        - Repair scope definition
        - Material requirements
        - Labor hours estimation
        - Special requirements (scaffolding, permits)

        Args:
            defect: Thermal defect data

        Returns:
            WorkScope with complete repair specifications
        """
        self._reset_calculation_steps()

        # Determine repair scope
        repair_scope = self._determine_repair_scope(defect)

        # Generate work description
        work_description = self._generate_work_description(defect, repair_scope)

        # Calculate material requirements
        materials = self._calculate_material_requirements(defect)

        # Calculate labor requirements
        labor = self._calculate_labor_requirements(defect, repair_scope)

        # Identify equipment requirements
        equipment = self._identify_equipment_requirements(defect)

        # Identify safety requirements
        safety = self._identify_safety_requirements(defect)

        # Identify permits
        permits = self._identify_permits(defect)

        # Calculate duration
        total_hours = sum(Decimal(str(l['hours'])) for l in labor)

        # Generate special instructions
        special_instructions = self._generate_special_instructions(defect)

        # Define quality checkpoints
        quality_checkpoints = self._define_quality_checkpoints(defect)

        return WorkScope(
            defect_id=defect.defect_id,
            repair_scope=repair_scope,
            work_description=work_description,
            material_requirements=materials,
            labor_requirements=labor,
            equipment_requirements=equipment,
            safety_requirements=safety,
            permits_required=permits,
            estimated_duration_hours=total_hours,
            special_instructions=special_instructions,
            quality_checkpoints=quality_checkpoints
        )

    def _generate_work_description(
        self,
        defect: ThermalDefect,
        scope: RepairScope
    ) -> str:
        """Generate work description text."""
        location = defect.location

        description_parts = [
            f"Repair {defect.damage_type.value.replace('_', ' ')} insulation",
            f"on {location.equipment_type.value.replace('_', ' ')} {location.equipment_tag}",
            f"in area {location.area_code}, unit {location.unit_code}.",
            f"Scope: {scope.value.replace('_', ' ')} ({float(defect.length_m):.1f}m).",
            f"Process temperature: {float(defect.process_temperature_c):.0f}C.",
            f"Insulation type: {defect.insulation_material.value.replace('_', ' ')},",
            f"thickness: {float(defect.insulation_thickness_mm):.0f}mm."
        ]

        return " ".join(description_parts)

    def _calculate_material_requirements(self, defect: ThermalDefect) -> List[Dict[str, Any]]:
        """Calculate material requirements."""
        materials = []
        length = float(defect.length_m)

        # Main insulation material
        material = defect.insulation_material.value
        thickness = int(defect.insulation_thickness_mm)

        # Add 15% waste factor
        insulation_length = length * 1.15

        materials.append({
            'item': f"{material.replace('_', ' ').title()} insulation",
            'thickness_mm': thickness,
            'quantity': round(insulation_length, 1),
            'unit': 'linear meters',
            'estimated_cost': float(self._estimate_material_cost(material, thickness, Decimal(str(insulation_length))))
        })

        # Jacketing
        jacket_area = length * 0.3 * 1.15  # Approximate circumference * length
        materials.append({
            'item': 'Aluminum jacketing 0.8mm',
            'quantity': round(jacket_area, 1),
            'unit': 'square meters',
            'estimated_cost': jacket_area * 25.0
        })

        # Bands and fasteners
        band_count = int(length / 0.3) + 2
        materials.append({
            'item': 'Stainless steel bands',
            'quantity': band_count,
            'unit': 'pieces',
            'estimated_cost': band_count * 3.0
        })

        # Sealant
        materials.append({
            'item': 'Silicone sealant',
            'quantity': max(1, int(length / 5)),
            'unit': 'tubes',
            'estimated_cost': max(1, int(length / 5)) * 15.0
        })

        # Vapor barrier (for cold systems)
        if float(defect.process_temperature_c) < float(defect.ambient_temperature_c):
            materials.append({
                'item': 'Vapor barrier tape',
                'quantity': round(insulation_length * 2, 1),
                'unit': 'linear meters',
                'estimated_cost': insulation_length * 2 * 5.0
            })

        return materials

    def _estimate_material_cost(
        self,
        material: str,
        thickness: int,
        length: Decimal
    ) -> float:
        """Estimate material cost."""
        cost_table = INSULATION_COST_PER_METER.get(material, {})

        if not cost_table:
            return float(length) * 30.0  # Default

        available_thicknesses = sorted(cost_table.keys())
        closest = min(available_thicknesses, key=lambda x: abs(x - thickness))

        return float(length) * cost_table[closest]

    def _calculate_labor_requirements(
        self,
        defect: ThermalDefect,
        scope: RepairScope
    ) -> List[Dict[str, Any]]:
        """Calculate labor requirements."""
        labor = []
        length = float(defect.length_m)

        # Access factor
        access_factor = 1 + (defect.location.access_difficulty - 1) * 0.2

        # Removal (if not missing)
        if defect.damage_type != DamageType.MISSING:
            removal_hours = (length / PRODUCTION_RATES['removal_damaged']) * access_factor
            labor.append({
                'trade': 'Journeyman Insulator',
                'task': 'Remove damaged insulation',
                'hours': round(removal_hours, 1),
                'rate': LABOR_RATES['journeyman_insulator']
            })

        # Surface preparation
        prep_hours = (length / PRODUCTION_RATES['surface_preparation']) * access_factor
        labor.append({
            'trade': 'Journeyman Insulator',
            'task': 'Surface preparation',
            'hours': round(prep_hours, 1),
            'rate': LABOR_RATES['journeyman_insulator']
        })

        # Installation
        if scope in {RepairScope.SPOT_REPAIR, RepairScope.SECTION_REPAIR}:
            install_rate = PRODUCTION_RATES['insulation_installation_complex']
        else:
            install_rate = PRODUCTION_RATES['insulation_installation_simple']

        install_hours = (length / install_rate) * access_factor
        labor.append({
            'trade': 'Journeyman Insulator',
            'task': 'Insulation installation',
            'hours': round(install_hours, 1),
            'rate': LABOR_RATES['journeyman_insulator']
        })

        # Jacketing
        jacket_hours = (length / PRODUCTION_RATES['jacketing_aluminum']) * access_factor
        labor.append({
            'trade': 'Journeyman Insulator',
            'task': 'Jacketing installation',
            'hours': round(jacket_hours, 1),
            'rate': LABOR_RATES['journeyman_insulator']
        })

        # Scaffolding if required
        if defect.location.scaffold_required:
            scaffold_hours = 4.0 + (length * 0.5)  # Setup plus linear
            labor.append({
                'trade': 'Scaffolding Worker',
                'task': 'Scaffold erection and dismantling',
                'hours': round(scaffold_hours, 1),
                'rate': LABOR_RATES['scaffolding_worker']
            })

        # Supervision
        total_hours = sum(l['hours'] for l in labor)
        supervision_hours = total_hours * 0.1
        labor.append({
            'trade': 'Supervisor',
            'task': 'Work supervision and quality control',
            'hours': round(supervision_hours, 1),
            'rate': LABOR_RATES['supervisor']
        })

        return labor

    def _identify_equipment_requirements(self, defect: ThermalDefect) -> List[str]:
        """Identify equipment requirements."""
        equipment = []

        # Standard equipment
        equipment.extend([
            "Insulation knife set",
            "Sheet metal tools",
            "Band tensioning tool",
            "Tape measure",
            "Marking tools"
        ])

        # Elevated work
        if defect.location.scaffold_required:
            equipment.append("Scaffold system")
        elif defect.location.elevation_m > 2.0:
            equipment.append("Mobile elevated work platform (MEWP)")

        # Access difficulty
        if defect.location.access_difficulty >= 3:
            equipment.append("Confined space entry equipment")

        # Pipe work
        if defect.location.equipment_type == EquipmentType.PIPE:
            equipment.append("Pipe circumference measuring tool")

        return equipment

    def _identify_safety_requirements(self, defect: ThermalDefect) -> List[str]:
        """Identify safety requirements."""
        safety = []

        # Standard PPE
        safety.append("Hard hat, safety glasses, gloves, safety boots")

        # Heat protection
        if float(defect.process_temperature_c) > 60:
            safety.append("Heat-resistant gloves")
            safety.append("Face shield for hot work")

        # Cold protection
        if float(defect.process_temperature_c) < 0:
            safety.append("Cryogenic gloves")
            safety.append("Face shield for cold work")

        # Elevated work
        if defect.location.elevation_m > 2.0 or defect.location.scaffold_required:
            safety.append("Fall protection harness")
            safety.append("Safety lanyard")

        # Respiratory protection
        if defect.insulation_material == InsulationMaterial.MINERAL_WOOL:
            safety.append("P2 dust mask")

        # Access difficulty
        if defect.location.access_difficulty >= 4:
            safety.append("Confined space entry procedures")
            safety.append("Gas detection equipment")
            safety.append("Rescue equipment standby")

        return safety

    def _identify_permits(self, defect: ThermalDefect) -> List[str]:
        """Identify required permits."""
        permits = []

        # Work permit
        permits.append("General work permit")

        # Hot work
        if float(defect.process_temperature_c) > 200:
            permits.append("Hot work permit")

        # Confined space
        if defect.location.access_difficulty >= 4:
            permits.append("Confined space entry permit")

        # Elevated work
        if defect.location.elevation_m > 2.0:
            permits.append("Working at height permit")

        # Process isolation
        if defect.location.process_isolation_required:
            permits.append("Process isolation permit (LOTO)")

        return permits

    def _generate_special_instructions(self, defect: ThermalDefect) -> List[str]:
        """Generate special instructions."""
        instructions = []

        # CUI inspection
        if defect.damage_type in {DamageType.CUI_SUSPECTED, DamageType.WET}:
            instructions.append(
                "CRITICAL: Perform visual inspection of pipe surface after insulation removal. "
                "Document any corrosion findings and report to engineering."
            )
            instructions.append("Do not proceed with re-insulation if significant corrosion found.")

        # Vapor barrier
        if float(defect.process_temperature_c) < float(defect.ambient_temperature_c):
            instructions.append(
                "Install vapor barrier on warm side of insulation. "
                "Ensure all joints are sealed to prevent moisture ingress."
            )

        # High temperature
        if float(defect.process_temperature_c) > 150:
            instructions.append(
                "Allow adequate cooling time before handling. "
                "Verify surface temperature < 50C before starting work."
            )

        # Expansion joints
        if float(defect.length_m) > 10:
            instructions.append("Install expansion joints at maximum 3m intervals.")

        return instructions

    def _define_quality_checkpoints(self, defect: ThermalDefect) -> List[str]:
        """Define quality control checkpoints."""
        checkpoints = [
            "Surface preparation - verify clean, dry, rust-free surface",
            "Insulation fit - verify tight fit with no gaps",
            "Joint sealing - verify all joints sealed",
            "Jacketing overlap - verify minimum 50mm overlap",
            "Band tension - verify proper band tension",
            "Final inspection - verify complete coverage and weatherproofing"
        ]

        # Cold systems
        if float(defect.process_temperature_c) < float(defect.ambient_temperature_c):
            checkpoints.insert(3, "Vapor barrier - verify continuous seal")

        # CUI-prone systems
        if 50 <= float(defect.process_temperature_c) <= 150:
            checkpoints.insert(1, "Surface condition - document and photograph pipe condition")

        return checkpoints

    # =========================================================================
    # 8. Create Repair Plan
    # =========================================================================

    def create_repair_plan(
        self,
        defects: List[ThermalDefect],
        budget: Optional[BudgetConstraint] = None
    ) -> OptimizedRepairPlan:
        """
        Create comprehensive optimized repair plan.

        Integrates all prioritization methodologies:
        - Multi-factor criticality scoring
        - ROI-based ranking
        - Risk assessment
        - Schedule optimization
        - Budget optimization
        - Work scope generation

        Args:
            defects: List of thermal defects to process
            budget: Optional budget constraint

        Returns:
            OptimizedRepairPlan with complete repair strategy
        """
        self._reset_calculation_steps()
        plan_start = datetime.now()

        # Step 1: Calculate criticality scores
        criticality_scores = [
            self.calculate_criticality_score(defect)
            for defect in defects
        ]

        # Step 2: Calculate ROI
        roi_results = [
            self.calculate_repair_roi(defect)
            for defect in defects
        ]

        # Step 3: Assess risks
        risk_assessments = [
            self.assess_risk_priority(defect)
            for defect in defects
        ]

        # Step 4: Optimize schedule
        scheduled_repairs = self.optimize_repair_schedule(
            defects,
            criticality_scores,
            roi_results,
            risk_assessments
        )

        # Step 5: Budget optimization if constraint provided
        if budget:
            selected_repairs, deferred_repairs = self.optimize_within_budget(
                scheduled_repairs,
                roi_results,
                budget
            )
        else:
            selected_repairs = scheduled_repairs
            deferred_repairs = []

        # Step 6: Generate work scopes for selected repairs
        work_scopes = [
            self.generate_work_scope(
                next(d for d in defects if d.defect_id == repair.defect_id)
            )
            for repair in selected_repairs
            if repair.priority_category != PriorityCategory.MONITOR
        ]

        # Categorize repairs
        emergency_repairs = [
            r for r in selected_repairs
            if r.priority_category == PriorityCategory.EMERGENCY
        ]
        urgent_repairs = [
            r for r in selected_repairs
            if r.priority_category == PriorityCategory.URGENT
        ]
        scheduled = [
            r for r in selected_repairs
            if r.priority_category in {PriorityCategory.HIGH, PriorityCategory.MEDIUM}
        ]

        # Calculate totals
        total_cost = sum(r.total_cost for r in selected_repairs)
        total_savings = sum(
            roi.annual_energy_savings
            for roi in roi_results
            if roi.defect_id in {r.defect_id for r in selected_repairs}
        )
        aggregate_npv = sum(
            roi.npv_over_life
            for roi in roi_results
            if roi.defect_id in {r.defect_id for r in selected_repairs}
            and roi.npv_over_life > Decimal("0")
        )

        # Budget utilization
        if budget:
            available = budget.total_budget * (
                Decimal("1") - budget.emergency_reserve_percent / Decimal("100")
            )
            utilization = (total_cost / available) * Decimal("100") if available > 0 else Decimal("0")
        else:
            utilization = Decimal("0")

        # Schedule summary
        schedule_summary = {
            'total_labor_hours': float(sum(r.labor_hours for r in selected_repairs)),
            'total_material_cost': float(sum(r.material_cost for r in selected_repairs)),
            'emergency_count': len(emergency_repairs),
            'urgent_count': len(urgent_repairs),
            'scheduled_count': len(scheduled),
            'deferred_count': len(deferred_repairs),
            'batch_count': len(set(r.batch_id for r in selected_repairs if r.batch_id))
        }

        # Calculate provenance hash
        provenance_data = {
            'defect_ids': [d.defect_id for d in defects],
            'criticality_scores': [float(c.composite_score) for c in criticality_scores],
            'total_cost': float(total_cost),
            'total_savings': float(total_savings),
            'generated_at': plan_start.isoformat()
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return OptimizedRepairPlan(
            plan_id=f"PLAN-{uuid.uuid4().hex[:8].upper()}",
            generated_at=plan_start,
            total_defects=len(defects),
            total_estimated_cost=self._apply_precision(total_cost, 2),
            total_annual_savings=self._apply_precision(total_savings, 2),
            aggregate_npv=self._apply_precision(aggregate_npv, 2),
            emergency_repairs=emergency_repairs,
            urgent_repairs=urgent_repairs,
            scheduled_repairs=scheduled,
            deferred_repairs=deferred_repairs,
            budget_utilization_percent=self._apply_precision(utilization, 1),
            work_scopes=work_scopes,
            schedule_summary=schedule_summary,
            provenance_hash=provenance_hash
        )


# =============================================================================
# Validation Functions
# =============================================================================

def validate_defect_input(defect: ThermalDefect) -> List[str]:
    """Validate thermal defect input data."""
    errors = []

    if float(defect.length_m) <= 0:
        errors.append("Length must be positive")

    if float(defect.length_m) > 1000:
        errors.append("Length exceeds maximum (1000m)")

    if float(defect.heat_loss_w_per_m) < 0:
        errors.append("Heat loss cannot be negative")

    if defect.existing_condition_score < 1 or defect.existing_condition_score > 10:
        errors.append("Condition score must be 1-10")

    return errors


def validate_criticality_weights(weights: CriticalityWeights) -> List[str]:
    """Validate criticality weights."""
    errors = []

    for weight_name in ['heat_loss_weight', 'safety_risk_weight', 'process_impact_weight',
                        'environmental_weight', 'asset_protection_weight']:
        weight = getattr(weights, weight_name)
        if weight < Decimal("0") or weight > Decimal("1"):
            errors.append(f"{weight_name} must be between 0 and 1")

    return errors


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main engine
    'RepairPrioritizationEngine',

    # Enumerations
    'PriorityCategory',
    'RiskLevel',
    'InsulationMaterial',
    'DamageType',
    'RepairScope',
    'EquipmentType',
    'OutageType',
    'SeverityRating',
    'OccurrenceRating',
    'DetectionRating',

    # Input models
    'DefectLocation',
    'ThermalDefect',
    'CriticalityWeights',
    'EconomicParameters',
    'ScheduleConstraints',
    'BudgetConstraint',

    # Output models
    'CriticalityScore',
    'RepairROI',
    'RiskAssessment',
    'ScheduledRepair',
    'WorkScope',
    'OptimizedRepairPlan',

    # Validation
    'validate_defect_input',
    'validate_criticality_weights',

    # Constants
    'HEAT_LOSS_SEVERITY_THRESHOLDS',
    'PERSONNEL_SAFETY_TEMP_C',
    'INSULATION_COST_PER_METER',
    'LABOR_RATES',
    'PRODUCTION_RATES',
    'CUI_RISK_FACTORS',
]
