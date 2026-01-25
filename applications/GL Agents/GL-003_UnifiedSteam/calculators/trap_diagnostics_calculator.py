"""
GL-003 UNIFIEDSTEAM - Steam Trap Diagnostics Calculator

Steam trap condition assessment and failure analysis.

Diagnostic Methods:
- Acoustic signature analysis (ultrasound)
- Temperature differential analysis
- Visual/operational indicators
- Statistical failure prediction

Failure Modes:
- BLOW_THROUGH: Trap fails open, steam passes continuously
- BLOCKED: Trap fails closed, condensate backs up
- LEAKING: Partial failure with intermittent steam loss
- NORMAL: Trap operating correctly

Reference: ASME FCI 69-1 Steam Trap Standard, Spirax Sarco Steam Trap Guide

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND STANDARDS
# =============================================================================

class TrapType(str, Enum):
    """Steam trap types."""
    THERMODYNAMIC = "THERMODYNAMIC"  # Disc trap
    THERMOSTATIC = "THERMOSTATIC"    # Bellows or bimetal
    MECHANICAL = "MECHANICAL"         # Float, bucket
    INVERTED_BUCKET = "INVERTED_BUCKET"
    FLOAT_THERMO = "FLOAT_THERMO"
    BIMETAL = "BIMETAL"
    BELLOWS = "BELLOWS"


class FailureMode(str, Enum):
    """Steam trap failure modes."""
    NORMAL = "NORMAL"
    BLOW_THROUGH = "BLOW_THROUGH"     # Fails open - continuous steam flow
    BLOCKED = "BLOCKED"               # Fails closed - no condensate discharge
    LEAKING = "LEAKING"               # Partial failure - intermittent steam loss
    COLD = "COLD"                     # No steam reaching trap (upstream issue)
    WATERLOGGED = "WATERLOGGED"       # Condensate not draining properly


class TrapCondition(str, Enum):
    """Overall trap condition classification."""
    GOOD = "GOOD"
    MARGINAL = "MARGINAL"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class MaintenancePriority(str, Enum):
    """Maintenance priority levels."""
    CRITICAL = "CRITICAL"     # Replace immediately
    HIGH = "HIGH"             # Replace within 1 week
    MEDIUM = "MEDIUM"         # Replace within 1 month
    LOW = "LOW"               # Schedule for next maintenance
    MONITOR = "MONITOR"       # Continue monitoring


# Typical steam loss rates by trap type and failure mode (kg/s at 1000 kPa)
# Based on industry data and FCI guidelines
TRAP_LOSS_RATES = {
    TrapType.THERMODYNAMIC: {
        FailureMode.BLOW_THROUGH: 0.030,  # kg/s at 1000 kPa
        FailureMode.LEAKING: 0.010,
        FailureMode.BLOCKED: 0.0,
    },
    TrapType.THERMOSTATIC: {
        FailureMode.BLOW_THROUGH: 0.025,
        FailureMode.LEAKING: 0.008,
        FailureMode.BLOCKED: 0.0,
    },
    TrapType.MECHANICAL: {
        FailureMode.BLOW_THROUGH: 0.040,
        FailureMode.LEAKING: 0.015,
        FailureMode.BLOCKED: 0.0,
    },
    TrapType.INVERTED_BUCKET: {
        FailureMode.BLOW_THROUGH: 0.035,
        FailureMode.LEAKING: 0.012,
        FailureMode.BLOCKED: 0.0,
    },
    TrapType.FLOAT_THERMO: {
        FailureMode.BLOW_THROUGH: 0.045,
        FailureMode.LEAKING: 0.018,
        FailureMode.BLOCKED: 0.0,
    },
}

# Acoustic frequency bands for trap analysis (Hz)
ACOUSTIC_BANDS = {
    "low": (20, 1000),
    "mid": (1000, 10000),
    "high": (10000, 40000),
    "ultrasonic": (40000, 100000),
}

# Acoustic signatures for different trap states
ACOUSTIC_SIGNATURES = {
    TrapCondition.GOOD: {
        "pattern": "intermittent",
        "ultrasonic_level_db": (30, 50),
        "cycling_detected": True,
    },
    TrapCondition.FAILED: {
        FailureMode.BLOW_THROUGH: {
            "pattern": "continuous",
            "ultrasonic_level_db": (60, 90),
            "cycling_detected": False,
        },
        FailureMode.BLOCKED: {
            "pattern": "silent",
            "ultrasonic_level_db": (0, 20),
            "cycling_detected": False,
        },
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrapInput:
    """Input parameters for trap diagnostics."""

    # Trap identification
    trap_tag: str
    trap_type: TrapType
    trap_size_mm: float = 15.0

    # Operating conditions
    operating_pressure_kpa: float
    inlet_temperature_c: float
    outlet_temperature_c: float

    # Acoustic data (optional)
    ultrasonic_level_db: Optional[float] = None
    acoustic_pattern: Optional[str] = None  # continuous, intermittent, silent
    cycling_detected: Optional[bool] = None

    # Installation
    installation_date: Optional[datetime] = None
    last_maintenance_date: Optional[datetime] = None
    operating_hours: Optional[float] = None

    # Process context
    application: str = "general"
    critical_service: bool = False


@dataclass
class TrapDiagnosticResult:
    """Result of trap condition analysis."""

    calculation_id: str
    timestamp: datetime

    # Trap identification
    trap_tag: str
    trap_type: TrapType

    # Condition assessment
    condition: TrapCondition
    failure_mode: FailureMode
    confidence_percent: float

    # Temperature analysis
    inlet_temp_c: float
    outlet_temp_c: float
    temp_differential_c: float
    expected_differential_c: float
    temp_diagnosis: str

    # Acoustic analysis
    acoustic_diagnosis: str
    ultrasonic_level_db: Optional[float]

    # Loss estimate
    estimated_loss_kg_s: float
    loss_severity: str

    # Recommendations
    recommended_action: str
    maintenance_priority: MaintenancePriority

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class FailurePrediction:
    """Failure probability prediction for a trap."""

    calculation_id: str
    timestamp: datetime

    # Trap identification
    trap_tag: str
    trap_type: TrapType

    # Failure probabilities
    failure_probability_30_days: float
    failure_probability_90_days: float
    failure_probability_365_days: float

    # Most likely failure mode
    predicted_failure_mode: FailureMode
    mode_probability: float

    # Risk factors
    age_factor: float
    operating_hours_factor: float
    pressure_factor: float
    application_factor: float
    overall_risk_score: float

    # Recommended inspection interval
    recommended_inspection_days: int

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class TrapEconomics:
    """Economic impact of trap condition."""

    calculation_id: str
    timestamp: datetime

    # Trap identification
    trap_tag: str
    condition: TrapCondition

    # Steam loss
    steam_loss_kg_s: float
    steam_loss_kg_hr: float
    steam_loss_kg_year: float

    # Energy loss
    energy_loss_kw: float
    energy_loss_mmbtu_hr: float

    # Economic impact
    hourly_cost: float
    daily_cost: float
    annual_cost: float

    # Repair economics
    estimated_repair_cost: float
    payback_days: float

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class MaintenancePriorityList:
    """Prioritized maintenance list for multiple traps."""

    calculation_id: str
    timestamp: datetime

    # Summary
    total_traps: int
    failed_traps: int
    marginal_traps: int
    good_traps: int

    # Total losses
    total_loss_kg_s: float
    total_annual_cost: float

    # Prioritized list
    priority_list: List[Dict[str, Any]]

    # Top actions
    critical_actions: List[str]
    high_priority_actions: List[str]

    # Provenance
    input_hash: str
    output_hash: str


# =============================================================================
# TRAP DIAGNOSTICS CALCULATOR
# =============================================================================

class TrapDiagnosticsCalculator:
    """
    Zero-hallucination steam trap diagnostics calculator.

    Implements deterministic diagnostics for:
    - Acoustic signature analysis
    - Temperature differential analysis
    - Failure mode classification
    - Loss rate estimation
    - Failure prediction
    - Maintenance prioritization

    All calculations use:
    - Deterministic classification rules
    - SHA-256 provenance hashing
    - Complete audit trails
    - NO LLM in diagnostic path

    Example:
        >>> calc = TrapDiagnosticsCalculator()
        >>> result = calc.analyze_trap_acoustics(
        ...     acoustic_features={"ultrasonic_db": 75, "pattern": "continuous"},
        ...     process_context={"trap_type": "THERMODYNAMIC", "pressure_kpa": 1000}
        ... )
        >>> print(f"Condition: {result.condition}")
    """

    VERSION = "1.0.0"
    FORMULA_VERSION = "TRAP_V1.0"

    def __init__(
        self,
        steam_cost_per_kg: float = 0.03,
        operating_hours_per_year: int = 8000,
        h_fg_reference_kj_kg: float = 2015.0,  # At ~1 MPa
    ) -> None:
        """
        Initialize trap diagnostics calculator.

        Args:
            steam_cost_per_kg: Cost of steam ($/kg)
            operating_hours_per_year: Annual operating hours
            h_fg_reference_kj_kg: Reference enthalpy of vaporization
        """
        self.steam_cost = steam_cost_per_kg
        self.operating_hours = operating_hours_per_year
        self.h_fg_ref = h_fg_reference_kj_kg

    def analyze_trap_acoustics(
        self,
        acoustic_features: Dict[str, Any],
        process_context: Dict[str, Any],
    ) -> TrapDiagnosticResult:
        """
        Analyze trap condition from acoustic features.

        DETERMINISTIC classification based on:
        - Ultrasonic sound level (dB)
        - Acoustic pattern (continuous, intermittent, silent)
        - Cycling detection

        Args:
            acoustic_features: Dict with ultrasonic_db, pattern, cycling_detected
            process_context: Dict with trap_type, pressure_kpa, temperatures

        Returns:
            TrapDiagnosticResult with condition assessment
        """
        # Extract features
        ultrasonic_db = acoustic_features.get("ultrasonic_db", 0)
        pattern = acoustic_features.get("pattern", "unknown")
        cycling = acoustic_features.get("cycling_detected", None)

        trap_type = TrapType(process_context.get("trap_type", "THERMODYNAMIC"))
        pressure_kpa = process_context.get("pressure_kpa", 1000)
        trap_tag = process_context.get("trap_tag", "UNKNOWN")
        inlet_temp = process_context.get("inlet_temp_c", 180)
        outlet_temp = process_context.get("outlet_temp_c", 100)

        # DETERMINISTIC CLASSIFICATION RULES

        # Rule 1: Continuous high ultrasonic = BLOW_THROUGH
        if pattern == "continuous" and ultrasonic_db > 55:
            condition = TrapCondition.FAILED
            failure_mode = FailureMode.BLOW_THROUGH
            confidence = min(95, 70 + (ultrasonic_db - 55) * 0.8)
            acoustic_diagnosis = f"High continuous ultrasonic ({ultrasonic_db} dB) indicates blow-through"

        # Rule 2: Silent = BLOCKED or COLD
        elif pattern == "silent" or ultrasonic_db < 15:
            condition = TrapCondition.FAILED
            # Differentiate BLOCKED vs COLD by inlet temperature
            if inlet_temp < 100:  # No steam reaching trap
                failure_mode = FailureMode.COLD
                acoustic_diagnosis = "Silent trap with low inlet temp - upstream issue"
            else:
                failure_mode = FailureMode.BLOCKED
                acoustic_diagnosis = f"Silent trap ({ultrasonic_db} dB) with steam present - blocked"
            confidence = 85

        # Rule 3: Intermittent with normal levels = GOOD
        elif pattern == "intermittent" and 25 <= ultrasonic_db <= 55:
            condition = TrapCondition.GOOD
            failure_mode = FailureMode.NORMAL
            confidence = 90 if cycling else 75
            acoustic_diagnosis = "Normal intermittent operation"

        # Rule 4: Moderate continuous = LEAKING
        elif pattern == "continuous" and 35 <= ultrasonic_db <= 55:
            condition = TrapCondition.MARGINAL
            failure_mode = FailureMode.LEAKING
            confidence = 70
            acoustic_diagnosis = f"Moderate continuous ultrasonic ({ultrasonic_db} dB) suggests leaking"

        # Rule 5: Uncertain - need more data
        else:
            condition = TrapCondition.UNKNOWN
            failure_mode = FailureMode.NORMAL
            confidence = 50
            acoustic_diagnosis = f"Inconclusive pattern ({pattern}, {ultrasonic_db} dB)"

        # Temperature analysis
        temp_diff = inlet_temp - outlet_temp
        expected_diff = self._expected_temp_differential(trap_type, pressure_kpa)

        if condition == TrapCondition.GOOD:
            temp_diagnosis = f"Normal subcooling ({temp_diff:.1f}C)"
        elif failure_mode == FailureMode.BLOW_THROUGH:
            if temp_diff < expected_diff * 0.5:
                temp_diagnosis = f"Low subcooling ({temp_diff:.1f}C) confirms blow-through"
                confidence = min(98, confidence + 5)
            else:
                temp_diagnosis = f"Subcooling ({temp_diff:.1f}C) inconsistent with acoustic"
                confidence = max(60, confidence - 10)
        elif failure_mode == FailureMode.BLOCKED:
            temp_diagnosis = f"High subcooling ({temp_diff:.1f}C) possible with blocked trap"
        else:
            temp_diagnosis = f"Subcooling: {temp_diff:.1f}C (expected ~{expected_diff:.1f}C)"

        # Estimate loss rate
        loss_rate = self.estimate_trap_loss_rate(trap_type, failure_mode, pressure_kpa)

        # Determine severity
        if loss_rate > 0.03:
            loss_severity = "CRITICAL"
        elif loss_rate > 0.015:
            loss_severity = "HIGH"
        elif loss_rate > 0.005:
            loss_severity = "MODERATE"
        else:
            loss_severity = "LOW"

        # Determine maintenance priority
        if failure_mode == FailureMode.BLOW_THROUGH:
            priority = MaintenancePriority.CRITICAL
            action = "Replace trap immediately - significant steam loss"
        elif failure_mode == FailureMode.BLOCKED:
            priority = MaintenancePriority.HIGH
            action = "Replace trap - condensate backup may cause water hammer"
        elif failure_mode == FailureMode.LEAKING:
            priority = MaintenancePriority.MEDIUM
            action = "Schedule trap replacement within 30 days"
        elif condition == TrapCondition.MARGINAL:
            priority = MaintenancePriority.MONITOR
            action = "Continue monitoring - retest in 30 days"
        else:
            priority = MaintenancePriority.LOW
            action = "Normal operation - routine inspection"

        # Compute hashes
        input_hash = self._compute_hash({
            "acoustic_features": acoustic_features,
            "process_context": process_context,
        })

        output_hash = self._compute_hash({
            "condition": condition.value,
            "failure_mode": failure_mode.value,
            "loss_rate": loss_rate,
        })

        return TrapDiagnosticResult(
            calculation_id=f"TRAPDIAG-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            trap_tag=trap_tag,
            trap_type=trap_type,
            condition=condition,
            failure_mode=failure_mode,
            confidence_percent=round(confidence, 1),
            inlet_temp_c=inlet_temp,
            outlet_temp_c=outlet_temp,
            temp_differential_c=round(temp_diff, 1),
            expected_differential_c=round(expected_diff, 1),
            temp_diagnosis=temp_diagnosis,
            acoustic_diagnosis=acoustic_diagnosis,
            ultrasonic_level_db=ultrasonic_db,
            estimated_loss_kg_s=round(loss_rate, 4),
            loss_severity=loss_severity,
            recommended_action=action,
            maintenance_priority=priority,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def estimate_trap_loss_rate(
        self,
        trap_type: TrapType,
        condition: FailureMode,
        operating_pressure_kpa: float,
    ) -> float:
        """
        Estimate steam loss rate for a failed trap.

        DETERMINISTIC calculation based on:
        - Trap type and orifice size
        - Failure mode
        - Operating pressure

        Loss scales with sqrt(pressure) from reference condition.

        Args:
            trap_type: Type of steam trap
            condition: Failure mode (BLOW_THROUGH, LEAKING, etc.)
            operating_pressure_kpa: Operating pressure (kPa)

        Returns:
            Estimated loss rate in kg/s
        """
        if condition == FailureMode.NORMAL:
            return 0.0

        if condition == FailureMode.BLOCKED:
            return 0.0

        # Get base loss rate at reference pressure (1000 kPa)
        trap_losses = TRAP_LOSS_RATES.get(trap_type, TRAP_LOSS_RATES[TrapType.THERMODYNAMIC])
        base_loss = trap_losses.get(condition, 0.01)

        # Scale with sqrt(pressure) - based on orifice flow
        # Reference pressure = 1000 kPa
        pressure_factor = math.sqrt(operating_pressure_kpa / 1000)

        # Calculate loss rate
        loss_rate = base_loss * pressure_factor

        return loss_rate

    def predict_failure_probability(
        self,
        trap_data: Dict[str, Any],
        maintenance_history: Optional[List[Dict]] = None,
    ) -> FailurePrediction:
        """
        Predict failure probability based on trap data and history.

        DETERMINISTIC prediction using:
        - Age-based reliability model
        - Operating hours
        - Pressure stress factor
        - Application severity
        - Historical failure patterns

        Args:
            trap_data: Current trap information
            maintenance_history: List of past maintenance events

        Returns:
            FailurePrediction with probability estimates
        """
        trap_tag = trap_data.get("trap_tag", "UNKNOWN")
        trap_type = TrapType(trap_data.get("trap_type", "THERMODYNAMIC"))

        # Extract factors
        age_years = trap_data.get("age_years", 2.0)
        operating_hours = trap_data.get("operating_hours", 10000)
        pressure_kpa = trap_data.get("operating_pressure_kpa", 1000)
        critical_service = trap_data.get("critical_service", False)
        application = trap_data.get("application", "general")

        # Age factor - exponential increase after 3 years
        # Base failure rate doubles every 3 years after initial period
        if age_years < 1:
            age_factor = 0.3
        elif age_years < 3:
            age_factor = 0.5 + (age_years - 1) * 0.25
        else:
            age_factor = 1.0 + (age_years - 3) * 0.4

        age_factor = min(3.0, age_factor)

        # Operating hours factor
        # Typical trap life: 30,000-50,000 hours
        expected_life_hours = self._expected_trap_life(trap_type)
        hours_factor = operating_hours / expected_life_hours
        hours_factor = min(2.0, hours_factor)

        # Pressure factor - higher pressure = more stress
        if pressure_kpa > 2000:
            pressure_factor = 1.5
        elif pressure_kpa > 1000:
            pressure_factor = 1.2
        else:
            pressure_factor = 1.0

        # Application factor
        app_factors = {
            "general": 1.0,
            "drip_leg": 0.8,
            "process": 1.2,
            "tracer": 1.3,
            "high_pressure": 1.5,
        }
        application_factor = app_factors.get(application, 1.0)

        if critical_service:
            application_factor *= 0.8  # More frequent maintenance

        # Overall risk score (0-100)
        risk_score = min(100, (
            age_factor * 20 +
            hours_factor * 25 +
            pressure_factor * 15 +
            application_factor * 15
        ))

        # Calculate failure probabilities
        # Base annual failure rate for trap type
        base_rates = {
            TrapType.THERMODYNAMIC: 0.15,
            TrapType.THERMOSTATIC: 0.10,
            TrapType.MECHANICAL: 0.08,
            TrapType.INVERTED_BUCKET: 0.07,
            TrapType.FLOAT_THERMO: 0.06,
        }
        base_annual_rate = base_rates.get(trap_type, 0.10)

        # Adjusted failure rate
        adjusted_rate = base_annual_rate * age_factor * hours_factor * pressure_factor * application_factor

        # Convert to time-specific probabilities
        # P(failure in t) = 1 - exp(-lambda * t)
        prob_30_days = 1 - math.exp(-adjusted_rate * 30 / 365)
        prob_90_days = 1 - math.exp(-adjusted_rate * 90 / 365)
        prob_365_days = 1 - math.exp(-adjusted_rate)

        # Predict most likely failure mode
        # Based on trap type
        mode_probs = self._failure_mode_distribution(trap_type, age_years)
        predicted_mode = max(mode_probs, key=mode_probs.get)
        mode_probability = mode_probs[predicted_mode]

        # Recommended inspection interval
        if risk_score > 70:
            inspection_days = 30
        elif risk_score > 50:
            inspection_days = 90
        elif risk_score > 30:
            inspection_days = 180
        else:
            inspection_days = 365

        # Compute hashes
        input_hash = self._compute_hash(trap_data)
        output_hash = self._compute_hash({
            "prob_30": prob_30_days,
            "prob_90": prob_90_days,
            "prob_365": prob_365_days,
        })

        return FailurePrediction(
            calculation_id=f"TRAPPRED-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            trap_tag=trap_tag,
            trap_type=trap_type,
            failure_probability_30_days=round(prob_30_days, 4),
            failure_probability_90_days=round(prob_90_days, 4),
            failure_probability_365_days=round(prob_365_days, 4),
            predicted_failure_mode=FailureMode(predicted_mode),
            mode_probability=round(mode_probability, 2),
            age_factor=round(age_factor, 2),
            operating_hours_factor=round(hours_factor, 2),
            pressure_factor=round(pressure_factor, 2),
            application_factor=round(application_factor, 2),
            overall_risk_score=round(risk_score, 1),
            recommended_inspection_days=inspection_days,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def classify_failure_mode(
        self,
        acoustic_signature: Dict[str, Any],
        temperature_differential: float,
        trap_type: Optional[TrapType] = None,
    ) -> Tuple[FailureMode, float]:
        """
        Classify failure mode from acoustic and temperature data.

        DETERMINISTIC classification using decision rules.

        Args:
            acoustic_signature: Dict with ultrasonic_db, pattern
            temperature_differential: Inlet - outlet temp (C)
            trap_type: Optional trap type for refined classification

        Returns:
            Tuple of (FailureMode, confidence)
        """
        ultrasonic_db = acoustic_signature.get("ultrasonic_db", 0)
        pattern = acoustic_signature.get("pattern", "unknown")

        # Decision tree classification
        if pattern == "continuous":
            if ultrasonic_db > 60:
                return FailureMode.BLOW_THROUGH, 0.95
            elif ultrasonic_db > 40:
                return FailureMode.LEAKING, 0.80
            else:
                return FailureMode.LEAKING, 0.60

        elif pattern == "silent":
            if temperature_differential > 50:
                return FailureMode.BLOCKED, 0.85
            else:
                return FailureMode.COLD, 0.75

        elif pattern == "intermittent":
            if 25 <= ultrasonic_db <= 55:
                return FailureMode.NORMAL, 0.90
            elif ultrasonic_db > 55:
                return FailureMode.LEAKING, 0.70
            else:
                return FailureMode.BLOCKED, 0.60

        else:
            return FailureMode.NORMAL, 0.50

    def prioritize_maintenance(
        self,
        trap_list: List[Dict[str, Any]],
        loss_rates: List[float],
        failure_probs: List[float],
    ) -> MaintenancePriorityList:
        """
        Generate prioritized maintenance list for multiple traps.

        Priority based on:
        - Current loss rate (economic impact)
        - Failure probability (risk)
        - Critical service designation

        Args:
            trap_list: List of trap data dictionaries
            loss_rates: Loss rate for each trap (kg/s)
            failure_probs: Failure probability for each trap

        Returns:
            MaintenancePriorityList with prioritized actions
        """
        # Validate inputs
        if len(trap_list) != len(loss_rates) or len(trap_list) != len(failure_probs):
            raise ValueError("Input lists must have same length")

        # Calculate priority score for each trap
        priority_items = []
        total_loss = 0.0
        total_cost = 0.0
        failed_count = 0
        marginal_count = 0
        good_count = 0

        for i, trap in enumerate(trap_list):
            loss_rate = loss_rates[i]
            fail_prob = failure_probs[i]

            # Economic loss
            annual_loss = loss_rate * 3600 * self.operating_hours * self.steam_cost
            total_cost += annual_loss
            total_loss += loss_rate

            # Determine condition
            if loss_rate > 0.015:
                condition = TrapCondition.FAILED
                failed_count += 1
            elif loss_rate > 0.005 or fail_prob > 0.5:
                condition = TrapCondition.MARGINAL
                marginal_count += 1
            else:
                condition = TrapCondition.GOOD
                good_count += 1

            # Priority score (higher = more urgent)
            # Weighted: 50% economic impact, 30% failure risk, 20% criticality
            economic_score = min(100, annual_loss / 100)  # $10,000/year = 100 points
            risk_score = fail_prob * 100
            critical_score = 100 if trap.get("critical_service", False) else 0

            priority_score = economic_score * 0.5 + risk_score * 0.3 + critical_score * 0.2

            # Determine priority level
            if priority_score > 70 or (loss_rate > 0.025):
                priority = MaintenancePriority.CRITICAL
            elif priority_score > 50 or (loss_rate > 0.015):
                priority = MaintenancePriority.HIGH
            elif priority_score > 30:
                priority = MaintenancePriority.MEDIUM
            elif priority_score > 15:
                priority = MaintenancePriority.LOW
            else:
                priority = MaintenancePriority.MONITOR

            priority_items.append({
                "trap_tag": trap.get("trap_tag", f"TRAP-{i}"),
                "trap_type": trap.get("trap_type", "UNKNOWN"),
                "condition": condition.value,
                "loss_rate_kg_s": round(loss_rate, 4),
                "annual_cost": round(annual_loss, 0),
                "failure_probability": round(fail_prob, 3),
                "priority_score": round(priority_score, 1),
                "priority": priority.value,
                "location": trap.get("location", "Unknown"),
            })

        # Sort by priority score (descending)
        priority_items.sort(key=lambda x: x["priority_score"], reverse=True)

        # Generate action lists
        critical_actions = []
        high_priority_actions = []

        for item in priority_items:
            if item["priority"] == MaintenancePriority.CRITICAL.value:
                critical_actions.append(
                    f"Replace {item['trap_tag']} immediately - "
                    f"${item['annual_cost']:,.0f}/year loss"
                )
            elif item["priority"] == MaintenancePriority.HIGH.value:
                high_priority_actions.append(
                    f"Schedule {item['trap_tag']} replacement within 1 week - "
                    f"${item['annual_cost']:,.0f}/year loss"
                )

        # Compute hashes
        input_hash = self._compute_hash({
            "trap_count": len(trap_list),
            "total_loss": total_loss,
        })

        output_hash = self._compute_hash({
            "failed_count": failed_count,
            "total_cost": total_cost,
        })

        return MaintenancePriorityList(
            calculation_id=f"TRAPPRIO-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            total_traps=len(trap_list),
            failed_traps=failed_count,
            marginal_traps=marginal_count,
            good_traps=good_count,
            total_loss_kg_s=round(total_loss, 4),
            total_annual_cost=round(total_cost, 0),
            priority_list=priority_items,
            critical_actions=critical_actions[:5],  # Top 5
            high_priority_actions=high_priority_actions[:5],
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def compute_trap_economics(
        self,
        trap_status: Dict[str, Any],
        steam_cost_per_kg: Optional[float] = None,
    ) -> TrapEconomics:
        """
        Calculate economic impact of trap condition.

        Args:
            trap_status: Dict with trap_tag, condition, loss_rate_kg_s
            steam_cost_per_kg: Optional override for steam cost

        Returns:
            TrapEconomics with cost analysis
        """
        steam_cost = steam_cost_per_kg or self.steam_cost

        trap_tag = trap_status.get("trap_tag", "UNKNOWN")
        condition = TrapCondition(trap_status.get("condition", "UNKNOWN"))
        loss_rate = trap_status.get("loss_rate_kg_s", 0.0)

        # Calculate losses
        loss_kg_hr = loss_rate * 3600
        loss_kg_year = loss_kg_hr * self.operating_hours

        # Energy loss
        energy_kw = loss_rate * self.h_fg_ref  # kW
        energy_mmbtu_hr = energy_kw * 3.412 / 1000  # Convert kW to MMBTU/hr

        # Economic impact
        hourly_cost = loss_kg_hr * steam_cost
        daily_cost = hourly_cost * 24
        annual_cost = loss_kg_year * steam_cost

        # Repair economics
        repair_costs = {
            TrapType.THERMODYNAMIC: 150,
            TrapType.THERMOSTATIC: 200,
            TrapType.MECHANICAL: 300,
            TrapType.INVERTED_BUCKET: 350,
            TrapType.FLOAT_THERMO: 400,
        }
        trap_type = TrapType(trap_status.get("trap_type", "THERMODYNAMIC"))
        repair_cost = repair_costs.get(trap_type, 250)

        # Payback period (days)
        if daily_cost > 0:
            payback_days = repair_cost / daily_cost
        else:
            payback_days = float("inf")

        # Compute hashes
        input_hash = self._compute_hash(trap_status)
        output_hash = self._compute_hash({
            "annual_cost": annual_cost,
            "payback_days": payback_days,
        })

        return TrapEconomics(
            calculation_id=f"TRAPECON-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            trap_tag=trap_tag,
            condition=condition,
            steam_loss_kg_s=round(loss_rate, 4),
            steam_loss_kg_hr=round(loss_kg_hr, 2),
            steam_loss_kg_year=round(loss_kg_year, 0),
            energy_loss_kw=round(energy_kw, 2),
            energy_loss_mmbtu_hr=round(energy_mmbtu_hr, 4),
            hourly_cost=round(hourly_cost, 2),
            daily_cost=round(daily_cost, 2),
            annual_cost=round(annual_cost, 0),
            estimated_repair_cost=repair_cost,
            payback_days=round(payback_days, 1) if payback_days != float("inf") else -1,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _expected_temp_differential(
        self,
        trap_type: TrapType,
        pressure_kpa: float,
    ) -> float:
        """
        Get expected temperature differential for healthy trap.

        Different trap types have different subcooling characteristics.
        """
        # Subcooling by trap type (degrees C below saturation)
        subcooling = {
            TrapType.THERMODYNAMIC: 2.0,
            TrapType.THERMOSTATIC: 15.0,
            TrapType.BIMETAL: 25.0,
            TrapType.BELLOWS: 12.0,
            TrapType.MECHANICAL: 3.0,
            TrapType.INVERTED_BUCKET: 2.0,
            TrapType.FLOAT_THERMO: 5.0,
        }

        return subcooling.get(trap_type, 10.0)

    def _expected_trap_life(self, trap_type: TrapType) -> float:
        """Get expected trap life in operating hours."""
        life_hours = {
            TrapType.THERMODYNAMIC: 20000,
            TrapType.THERMOSTATIC: 30000,
            TrapType.MECHANICAL: 40000,
            TrapType.INVERTED_BUCKET: 50000,
            TrapType.FLOAT_THERMO: 45000,
            TrapType.BIMETAL: 35000,
            TrapType.BELLOWS: 25000,
        }
        return life_hours.get(trap_type, 30000)

    def _failure_mode_distribution(
        self,
        trap_type: TrapType,
        age_years: float,
    ) -> Dict[str, float]:
        """
        Get failure mode probability distribution for trap type.

        Returns dict of failure mode -> probability.
        """
        # Base distributions by trap type
        if trap_type in [TrapType.THERMODYNAMIC]:
            # Disc traps tend to fail open
            base = {
                "BLOW_THROUGH": 0.60,
                "BLOCKED": 0.15,
                "LEAKING": 0.25,
            }
        elif trap_type in [TrapType.THERMOSTATIC, TrapType.BIMETAL, TrapType.BELLOWS]:
            # Thermostatic traps can fail either way
            base = {
                "BLOW_THROUGH": 0.40,
                "BLOCKED": 0.30,
                "LEAKING": 0.30,
            }
        else:  # Mechanical traps
            # Float/bucket traps tend to fail closed
            base = {
                "BLOW_THROUGH": 0.25,
                "BLOCKED": 0.50,
                "LEAKING": 0.25,
            }

        # Age adjustment - older traps more likely to fail open
        if age_years > 5:
            base["BLOW_THROUGH"] *= 1.2
            base["BLOCKED"] *= 0.9

        # Normalize
        total = sum(base.values())
        return {k: v / total for k, v in base.items()}

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
