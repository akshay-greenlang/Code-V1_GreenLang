"""
Unit Tests: Steam Trap Diagnostics

Tests the steam trap diagnostics module including:
- Failure mode classification
- Loss rate estimation
- Maintenance prioritization

Reference: ASME PTC 39, Steam Trap Handbook

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
from datetime import datetime, timedelta
import hashlib
import json


# =============================================================================
# Data Classes and Enumerations
# =============================================================================

class TrapType(Enum):
    """Types of steam traps."""
    THERMOSTATIC = auto()
    THERMODYNAMIC = auto()
    MECHANICAL_FLOAT = auto()
    MECHANICAL_BUCKET = auto()
    ORIFICE = auto()


class TrapFailureMode(Enum):
    """Steam trap failure modes."""
    NORMAL = auto()            # Operating correctly
    FAILED_OPEN = auto()       # Stuck open - leaking steam
    FAILED_CLOSED = auto()     # Stuck closed - not draining condensate
    BLOW_THROUGH = auto()      # Intermittent steam blow-through
    PARTIAL_BLOCKAGE = auto()  # Partially blocked
    INTERNAL_EROSION = auto()  # Internal wear/erosion
    UNKNOWN = auto()


class MaintenancePriority(Enum):
    """Maintenance priority levels."""
    CRITICAL = auto()    # Immediate attention required
    HIGH = auto()        # Schedule within 1 week
    MEDIUM = auto()      # Schedule within 1 month
    LOW = auto()         # Monitor, schedule when convenient
    NONE = auto()        # No action required


class DiagnosticMethod(Enum):
    """Diagnostic method used."""
    ACOUSTIC = auto()
    TEMPERATURE = auto()
    VISUAL = auto()
    COMBINED = auto()


@dataclass
class TrapData:
    """Steam trap operating data."""
    trap_id: str
    trap_type: TrapType
    location: str
    # Operating conditions
    inlet_pressure_mpa: float
    outlet_pressure_mpa: float
    inlet_temperature_k: float
    outlet_temperature_k: float
    # Acoustic data
    ultrasonic_db: Optional[float] = None
    acoustic_frequency_hz: Optional[float] = None
    cycle_rate_per_min: Optional[float] = None
    # Installation info
    orifice_diameter_mm: float = 10.0
    install_date: Optional[datetime] = None
    last_maintenance: Optional[datetime] = None


@dataclass
class DiagnosticResult:
    """Result of trap diagnostic analysis."""
    trap_id: str
    failure_mode: TrapFailureMode
    confidence: float
    diagnostic_method: DiagnosticMethod
    steam_loss_kg_s: float
    energy_loss_kw: float
    annual_loss_cost: float
    maintenance_priority: MaintenancePriority
    explanation: str
    recommendations: List[str]
    provenance_hash: str


@dataclass
class FleetDiagnostics:
    """Diagnostics summary for fleet of traps."""
    total_traps: int
    healthy_traps: int
    failed_traps: int
    failure_rate_percent: float
    total_steam_loss_kg_s: float
    total_energy_loss_kw: float
    total_annual_cost: float
    priority_breakdown: Dict[MaintenancePriority, int]
    failure_mode_breakdown: Dict[TrapFailureMode, int]
    top_priority_traps: List[str]
    provenance_hash: str


# =============================================================================
# Constants
# =============================================================================

# Economic parameters
STEAM_COST_PER_TON = 30.0  # Currency per ton of steam
OPERATING_HOURS_PER_YEAR = 8000

# Diagnostic thresholds
ULTRASONIC_NORMAL_MAX_DB = 70
ULTRASONIC_WARNING_DB = 85
ULTRASONIC_FAILED_DB = 95

TEMPERATURE_DROP_NORMAL_MIN_K = 5
TEMPERATURE_DROP_SUBCOOLED_MAX_K = 30

# Steam properties for loss calculation
STEAM_DENSITY_AT_1MPA = 5.145  # kg/m3
LATENT_HEAT_AT_1MPA = 2015.0   # kJ/kg


# =============================================================================
# Steam Trap Diagnostics Implementation
# =============================================================================

class TrapDiagnosticError(Exception):
    """Error in trap diagnostic calculation."""
    pass


def get_saturation_temperature(pressure_mpa: float) -> float:
    """Calculate saturation temperature from pressure."""
    if pressure_mpa < 0.001 or pressure_mpa > 22.064:
        raise TrapDiagnosticError(f"Pressure {pressure_mpa} MPa outside valid range")

    n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
         0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
         -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
         0.65017534844798e3]

    beta = pressure_mpa ** 0.25
    E = beta ** 2 + n[2] * beta + n[5]
    F = n[0] * beta ** 2 + n[3] * beta + n[6]
    G = n[1] * beta ** 2 + n[4] * beta + n[7]
    D = 2 * G / (-F - math.sqrt(F ** 2 - 4 * E * G))

    return (n[9] + D - math.sqrt((n[9] + D) ** 2 - 4 * (n[8] + n[9] * D))) / 2


def get_latent_heat(pressure_mpa: float) -> float:
    """Calculate latent heat at given pressure."""
    t_sat = get_saturation_temperature(pressure_mpa)
    t_crit = 647.096
    h_fg_ref = 2257.0

    if t_sat >= t_crit:
        return 0.0

    tr = t_sat / t_crit
    return h_fg_ref * ((1 - tr) / (1 - 373.15 / t_crit)) ** 0.38


def classify_failure_mode_acoustic(
    ultrasonic_db: float,
    acoustic_frequency_hz: Optional[float],
    cycle_rate: Optional[float],
    trap_type: TrapType
) -> Tuple[TrapFailureMode, float]:
    """
    Classify failure mode based on acoustic signature.

    Returns: (failure_mode, confidence)
    """
    # High ultrasonic indicates steam leakage
    if ultrasonic_db > ULTRASONIC_FAILED_DB:
        return TrapFailureMode.FAILED_OPEN, 0.9

    if ultrasonic_db > ULTRASONIC_WARNING_DB:
        # Could be blow-through or partial failure
        if cycle_rate is not None:
            if trap_type == TrapType.THERMODYNAMIC and cycle_rate > 30:
                return TrapFailureMode.BLOW_THROUGH, 0.8
            elif trap_type == TrapType.MECHANICAL_BUCKET and cycle_rate < 1:
                return TrapFailureMode.FAILED_CLOSED, 0.7
        return TrapFailureMode.BLOW_THROUGH, 0.6

    if ultrasonic_db < 50:
        # Very low acoustic - may be failed closed
        if trap_type in [TrapType.MECHANICAL_FLOAT, TrapType.MECHANICAL_BUCKET]:
            return TrapFailureMode.FAILED_CLOSED, 0.5
        return TrapFailureMode.UNKNOWN, 0.4

    # Normal range
    return TrapFailureMode.NORMAL, 0.85


def classify_failure_mode_temperature(
    inlet_temp_k: float,
    outlet_temp_k: float,
    inlet_pressure_mpa: float,
    outlet_pressure_mpa: float
) -> Tuple[TrapFailureMode, float]:
    """
    Classify failure mode based on temperature differential.

    Returns: (failure_mode, confidence)
    """
    # Get saturation temperatures
    t_sat_in = get_saturation_temperature(inlet_pressure_mpa)

    # Check if inlet is at saturation
    inlet_subcooling = t_sat_in - inlet_temp_k

    # Temperature drop across trap
    temp_drop = inlet_temp_k - outlet_temp_k

    # Failed open: outlet near inlet temperature (steam passing through)
    if temp_drop < 2.0:
        if inlet_subcooling < 5:  # Inlet at saturation
            return TrapFailureMode.FAILED_OPEN, 0.85
        return TrapFailureMode.UNKNOWN, 0.4

    # Failed closed: large temperature drop (condensate backing up)
    if temp_drop > TEMPERATURE_DROP_SUBCOOLED_MAX_K:
        return TrapFailureMode.FAILED_CLOSED, 0.7

    # Normal: moderate subcooling
    if TEMPERATURE_DROP_NORMAL_MIN_K < temp_drop < TEMPERATURE_DROP_SUBCOOLED_MAX_K:
        return TrapFailureMode.NORMAL, 0.75

    return TrapFailureMode.UNKNOWN, 0.3


def classify_failure_mode(trap_data: TrapData) -> Tuple[TrapFailureMode, float, DiagnosticMethod]:
    """
    Classify failure mode using available diagnostic data.

    Combines acoustic and temperature methods when both available.
    """
    acoustic_available = trap_data.ultrasonic_db is not None
    temp_available = trap_data.inlet_temperature_k > 0 and trap_data.outlet_temperature_k > 0

    if acoustic_available and temp_available:
        # Combined diagnosis
        mode_acoustic, conf_acoustic = classify_failure_mode_acoustic(
            trap_data.ultrasonic_db,
            trap_data.acoustic_frequency_hz,
            trap_data.cycle_rate_per_min,
            trap_data.trap_type
        )

        mode_temp, conf_temp = classify_failure_mode_temperature(
            trap_data.inlet_temperature_k,
            trap_data.outlet_temperature_k,
            trap_data.inlet_pressure_mpa,
            trap_data.outlet_pressure_mpa
        )

        # If methods agree, increase confidence
        if mode_acoustic == mode_temp:
            return mode_acoustic, min(0.95, conf_acoustic + conf_temp * 0.5), DiagnosticMethod.COMBINED

        # If disagree, use higher confidence method
        if conf_acoustic > conf_temp:
            return mode_acoustic, conf_acoustic * 0.8, DiagnosticMethod.ACOUSTIC
        else:
            return mode_temp, conf_temp * 0.8, DiagnosticMethod.TEMPERATURE

    elif acoustic_available:
        mode, conf = classify_failure_mode_acoustic(
            trap_data.ultrasonic_db,
            trap_data.acoustic_frequency_hz,
            trap_data.cycle_rate_per_min,
            trap_data.trap_type
        )
        return mode, conf, DiagnosticMethod.ACOUSTIC

    elif temp_available:
        mode, conf = classify_failure_mode_temperature(
            trap_data.inlet_temperature_k,
            trap_data.outlet_temperature_k,
            trap_data.inlet_pressure_mpa,
            trap_data.outlet_pressure_mpa
        )
        return mode, conf, DiagnosticMethod.TEMPERATURE

    else:
        return TrapFailureMode.UNKNOWN, 0.0, DiagnosticMethod.VISUAL


def estimate_steam_loss(
    failure_mode: TrapFailureMode,
    inlet_pressure_mpa: float,
    orifice_diameter_mm: float
) -> float:
    """
    Estimate steam loss rate based on failure mode and orifice size.

    Uses simplified orifice flow equation for failed-open traps.

    Returns: Steam loss in kg/s
    """
    if failure_mode == TrapFailureMode.NORMAL:
        return 0.0

    if failure_mode == TrapFailureMode.FAILED_CLOSED:
        return 0.0  # No steam loss, but condensate backup

    # Orifice area
    orifice_area_m2 = math.pi * (orifice_diameter_mm / 1000 / 2) ** 2

    # Discharge coefficient (typical for sharp-edged orifice)
    cd = 0.62

    # Steam density at operating pressure (simplified)
    # Using ideal gas approximation: rho = P / (R * T)
    t_sat = get_saturation_temperature(inlet_pressure_mpa)
    R_steam = 461.5  # J/(kg.K)
    steam_density = (inlet_pressure_mpa * 1e6) / (R_steam * t_sat)

    # Pressure ratio for choked flow assumption
    # For failed-open, assume choked flow to atmosphere

    if failure_mode == TrapFailureMode.FAILED_OPEN:
        # Full flow through orifice
        # m_dot = Cd * A * sqrt(2 * rho * dP)
        dp = inlet_pressure_mpa * 1e6 - 101325  # Pa
        if dp <= 0:
            dp = 1000  # Minimum pressure differential

        mass_flow = cd * orifice_area_m2 * math.sqrt(2 * steam_density * dp)
        return mass_flow

    elif failure_mode == TrapFailureMode.BLOW_THROUGH:
        # Intermittent loss - assume 30% duty cycle
        dp = inlet_pressure_mpa * 1e6 - 101325
        if dp <= 0:
            dp = 1000

        full_flow = cd * orifice_area_m2 * math.sqrt(2 * steam_density * dp)
        return full_flow * 0.3

    elif failure_mode == TrapFailureMode.INTERNAL_EROSION:
        # Partial leakage - assume 10% of full flow
        dp = inlet_pressure_mpa * 1e6 - 101325
        if dp <= 0:
            dp = 1000

        full_flow = cd * orifice_area_m2 * math.sqrt(2 * steam_density * dp)
        return full_flow * 0.1

    return 0.0


def estimate_energy_loss(steam_loss_kg_s: float, pressure_mpa: float) -> float:
    """
    Estimate energy loss from steam leakage.

    Returns: Energy loss in kW
    """
    if steam_loss_kg_s <= 0:
        return 0.0

    # Get latent heat at operating pressure
    h_fg = get_latent_heat(pressure_mpa)

    return steam_loss_kg_s * h_fg


def estimate_annual_cost(
    steam_loss_kg_s: float,
    operating_hours: float = OPERATING_HOURS_PER_YEAR,
    steam_cost_per_ton: float = STEAM_COST_PER_TON
) -> float:
    """
    Estimate annual cost of steam loss.

    Returns: Annual cost in currency units
    """
    if steam_loss_kg_s <= 0:
        return 0.0

    # Convert to tons per hour
    tons_per_hour = steam_loss_kg_s * 3.6  # kg/s to t/h

    return tons_per_hour * operating_hours * steam_cost_per_ton


def determine_maintenance_priority(
    failure_mode: TrapFailureMode,
    energy_loss_kw: float,
    annual_cost: float,
    confidence: float
) -> MaintenancePriority:
    """
    Determine maintenance priority based on failure severity and losses.
    """
    if failure_mode == TrapFailureMode.NORMAL:
        return MaintenancePriority.NONE

    if failure_mode == TrapFailureMode.UNKNOWN:
        if confidence < 0.5:
            return MaintenancePriority.LOW  # Need more investigation
        return MaintenancePriority.MEDIUM

    # Failed open is most urgent due to continuous steam loss
    if failure_mode == TrapFailureMode.FAILED_OPEN:
        if energy_loss_kw > 50:
            return MaintenancePriority.CRITICAL
        elif energy_loss_kw > 20:
            return MaintenancePriority.HIGH
        return MaintenancePriority.MEDIUM

    # Failed closed can cause water hammer - also urgent
    if failure_mode == TrapFailureMode.FAILED_CLOSED:
        return MaintenancePriority.HIGH

    # Blow-through and partial failures
    if failure_mode in [TrapFailureMode.BLOW_THROUGH, TrapFailureMode.INTERNAL_EROSION]:
        if annual_cost > 5000:
            return MaintenancePriority.HIGH
        elif annual_cost > 1000:
            return MaintenancePriority.MEDIUM
        return MaintenancePriority.LOW

    return MaintenancePriority.LOW


def generate_recommendations(
    trap_data: TrapData,
    failure_mode: TrapFailureMode,
    confidence: float
) -> List[str]:
    """
    Generate maintenance recommendations based on diagnosis.
    """
    recommendations = []

    if failure_mode == TrapFailureMode.NORMAL:
        recommendations.append("No immediate action required. Continue routine monitoring.")
        return recommendations

    if failure_mode == TrapFailureMode.UNKNOWN:
        recommendations.append("Unable to determine trap status. Perform manual inspection.")
        if confidence < 0.5:
            recommendations.append("Consider additional diagnostic testing (visual, temperature).")
        return recommendations

    if failure_mode == TrapFailureMode.FAILED_OPEN:
        recommendations.append("Replace trap immediately to stop steam loss.")
        recommendations.append(f"Consider upgrading to more robust trap type for this application.")

        if trap_data.trap_type == TrapType.THERMODYNAMIC:
            recommendations.append("Check for contamination that may prevent disc seating.")
        elif trap_data.trap_type in [TrapType.MECHANICAL_FLOAT, TrapType.MECHANICAL_BUCKET]:
            recommendations.append("Check for internal corrosion or mechanical damage.")

    elif failure_mode == TrapFailureMode.FAILED_CLOSED:
        recommendations.append("Inspect and repair/replace trap to restore condensate drainage.")
        recommendations.append("Warning: Failed closed traps can cause water hammer.")

        if trap_data.trap_type == TrapType.THERMOSTATIC:
            recommendations.append("Check bellows or bimetallic element for failure.")
        elif trap_data.trap_type == TrapType.THERMODYNAMIC:
            recommendations.append("Check for debris blocking orifice.")

    elif failure_mode == TrapFailureMode.BLOW_THROUGH:
        recommendations.append("Trap may be oversized or operating outside design conditions.")
        recommendations.append("Verify trap sizing for current load conditions.")
        recommendations.append("Consider installing check valve if reverse flow possible.")

    elif failure_mode == TrapFailureMode.INTERNAL_EROSION:
        recommendations.append("Schedule trap replacement during next maintenance window.")
        recommendations.append("Consider installing strainer upstream to reduce wear.")

    # Age-based recommendations
    if trap_data.install_date:
        age_years = (datetime.now() - trap_data.install_date).days / 365
        if age_years > 5:
            recommendations.append(f"Trap is {age_years:.1f} years old. Consider proactive replacement.")

    return recommendations


def diagnose_trap(trap_data: TrapData) -> DiagnosticResult:
    """
    Perform complete diagnostic analysis on a steam trap.
    """
    # Classify failure mode
    failure_mode, confidence, method = classify_failure_mode(trap_data)

    # Estimate losses
    steam_loss = estimate_steam_loss(
        failure_mode,
        trap_data.inlet_pressure_mpa,
        trap_data.orifice_diameter_mm
    )
    energy_loss = estimate_energy_loss(steam_loss, trap_data.inlet_pressure_mpa)
    annual_cost = estimate_annual_cost(steam_loss)

    # Determine priority
    priority = determine_maintenance_priority(
        failure_mode, energy_loss, annual_cost, confidence
    )

    # Generate recommendations
    recommendations = generate_recommendations(trap_data, failure_mode, confidence)

    # Generate explanation
    if failure_mode == TrapFailureMode.NORMAL:
        explanation = "Trap is operating normally with no detected issues."
    elif failure_mode == TrapFailureMode.FAILED_OPEN:
        explanation = f"Trap is stuck open, allowing live steam to pass. Estimated loss: {steam_loss*3600:.1f} kg/h."
    elif failure_mode == TrapFailureMode.FAILED_CLOSED:
        explanation = "Trap is stuck closed, preventing condensate drainage. Risk of water hammer."
    elif failure_mode == TrapFailureMode.BLOW_THROUGH:
        explanation = "Trap is experiencing intermittent steam blow-through."
    else:
        explanation = f"Trap status: {failure_mode.name}. Confidence: {confidence:.0%}."

    # Provenance hash
    provenance_hash = hashlib.sha256(
        json.dumps({
            "trap_id": trap_data.trap_id,
            "failure_mode": failure_mode.name,
            "steam_loss": steam_loss,
            "confidence": confidence
        }, sort_keys=True).encode()
    ).hexdigest()

    return DiagnosticResult(
        trap_id=trap_data.trap_id,
        failure_mode=failure_mode,
        confidence=confidence,
        diagnostic_method=method,
        steam_loss_kg_s=steam_loss,
        energy_loss_kw=energy_loss,
        annual_loss_cost=annual_cost,
        maintenance_priority=priority,
        explanation=explanation,
        recommendations=recommendations,
        provenance_hash=provenance_hash
    )


def diagnose_fleet(traps: List[TrapData]) -> FleetDiagnostics:
    """
    Perform diagnostic analysis on fleet of steam traps.
    """
    if not traps:
        raise TrapDiagnosticError("No traps provided for fleet diagnosis")

    results = [diagnose_trap(trap) for trap in traps]

    # Count failures
    healthy = sum(1 for r in results if r.failure_mode == TrapFailureMode.NORMAL)
    failed = sum(1 for r in results if r.failure_mode not in [TrapFailureMode.NORMAL, TrapFailureMode.UNKNOWN])

    # Sum losses
    total_steam_loss = sum(r.steam_loss_kg_s for r in results)
    total_energy_loss = sum(r.energy_loss_kw for r in results)
    total_annual_cost = sum(r.annual_loss_cost for r in results)

    # Priority breakdown
    priority_breakdown = {p: 0 for p in MaintenancePriority}
    for r in results:
        priority_breakdown[r.maintenance_priority] += 1

    # Failure mode breakdown
    mode_breakdown = {m: 0 for m in TrapFailureMode}
    for r in results:
        mode_breakdown[r.failure_mode] += 1

    # Top priority traps
    critical_traps = [r.trap_id for r in results if r.maintenance_priority == MaintenancePriority.CRITICAL]
    high_traps = [r.trap_id for r in results if r.maintenance_priority == MaintenancePriority.HIGH]
    top_priority = (critical_traps + high_traps)[:10]  # Top 10

    provenance_hash = hashlib.sha256(
        json.dumps({
            "total_traps": len(traps),
            "healthy": healthy,
            "total_steam_loss": total_steam_loss
        }, sort_keys=True).encode()
    ).hexdigest()

    return FleetDiagnostics(
        total_traps=len(traps),
        healthy_traps=healthy,
        failed_traps=failed,
        failure_rate_percent=failed / len(traps) * 100 if traps else 0,
        total_steam_loss_kg_s=total_steam_loss,
        total_energy_loss_kw=total_energy_loss,
        total_annual_cost=total_annual_cost,
        priority_breakdown=priority_breakdown,
        failure_mode_breakdown=mode_breakdown,
        top_priority_traps=top_priority,
        provenance_hash=provenance_hash
    )


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def healthy_trap():
    """Healthy steam trap data."""
    return TrapData(
        trap_id="TRAP-001",
        trap_type=TrapType.THERMODYNAMIC,
        location="Building A",
        inlet_pressure_mpa=1.0,
        outlet_pressure_mpa=0.1,
        inlet_temperature_k=453.0,
        outlet_temperature_k=440.0,
        ultrasonic_db=65,
        acoustic_frequency_hz=None,
        cycle_rate_per_min=10,
        orifice_diameter_mm=10.0
    )


@pytest.fixture
def failed_open_trap():
    """Failed open (leaking) trap data."""
    return TrapData(
        trap_id="TRAP-002",
        trap_type=TrapType.THERMODYNAMIC,
        location="Building B",
        inlet_pressure_mpa=1.0,
        outlet_pressure_mpa=0.1,
        inlet_temperature_k=453.0,
        outlet_temperature_k=451.0,  # Near inlet temp - steam passing
        ultrasonic_db=98,  # High ultrasonic
        acoustic_frequency_hz=None,
        cycle_rate_per_min=0,
        orifice_diameter_mm=10.0
    )


@pytest.fixture
def failed_closed_trap():
    """Failed closed trap data."""
    return TrapData(
        trap_id="TRAP-003",
        trap_type=TrapType.MECHANICAL_FLOAT,
        location="Building C",
        inlet_pressure_mpa=1.0,
        outlet_pressure_mpa=0.1,
        inlet_temperature_k=453.0,
        outlet_temperature_k=400.0,  # Large subcooling - condensate backing up
        ultrasonic_db=40,  # Low ultrasonic
        acoustic_frequency_hz=None,
        cycle_rate_per_min=0,
        orifice_diameter_mm=15.0
    )


@pytest.fixture
def trap_fleet():
    """Fleet of mixed-condition traps."""
    return [
        TrapData("TRAP-H1", TrapType.THERMODYNAMIC, "Area 1", 1.0, 0.1, 453, 440, 65, None, 10, 10),
        TrapData("TRAP-H2", TrapType.MECHANICAL_FLOAT, "Area 1", 1.0, 0.1, 453, 438, 68, None, 8, 12),
        TrapData("TRAP-H3", TrapType.THERMOSTATIC, "Area 2", 0.8, 0.1, 443, 430, 62, None, None, 8),
        TrapData("TRAP-F1", TrapType.THERMODYNAMIC, "Area 2", 1.0, 0.1, 453, 451, 96, None, 0, 10),  # Failed open
        TrapData("TRAP-F2", TrapType.MECHANICAL_BUCKET, "Area 3", 1.0, 0.1, 453, 395, 45, None, 0, 15),  # Failed closed
        TrapData("TRAP-B1", TrapType.THERMODYNAMIC, "Area 3", 1.0, 0.1, 453, 448, 88, None, 45, 10),  # Blow-through
    ]


# =============================================================================
# Test Classes
# =============================================================================

class TestFailureModeClassification:
    """Test failure mode classification logic."""

    def test_classify_healthy_trap_acoustic(self):
        """Test classification of healthy trap from acoustic data."""
        mode, confidence = classify_failure_mode_acoustic(
            ultrasonic_db=65,
            acoustic_frequency_hz=None,
            cycle_rate=10,
            trap_type=TrapType.THERMODYNAMIC
        )
        assert mode == TrapFailureMode.NORMAL
        assert confidence >= 0.8

    def test_classify_failed_open_acoustic(self):
        """Test classification of failed open trap from acoustic data."""
        mode, confidence = classify_failure_mode_acoustic(
            ultrasonic_db=98,
            acoustic_frequency_hz=None,
            cycle_rate=0,
            trap_type=TrapType.THERMODYNAMIC
        )
        assert mode == TrapFailureMode.FAILED_OPEN
        assert confidence >= 0.8

    def test_classify_blow_through_acoustic(self):
        """Test classification of blow-through from acoustic data."""
        mode, confidence = classify_failure_mode_acoustic(
            ultrasonic_db=88,
            acoustic_frequency_hz=None,
            cycle_rate=35,
            trap_type=TrapType.THERMODYNAMIC
        )
        assert mode == TrapFailureMode.BLOW_THROUGH

    def test_classify_failed_open_temperature(self):
        """Test classification of failed open from temperature."""
        mode, confidence = classify_failure_mode_temperature(
            inlet_temp_k=453.0,
            outlet_temp_k=451.0,  # Only 2 K drop
            inlet_pressure_mpa=1.0,
            outlet_pressure_mpa=0.1
        )
        assert mode == TrapFailureMode.FAILED_OPEN

    def test_classify_failed_closed_temperature(self):
        """Test classification of failed closed from temperature."""
        mode, confidence = classify_failure_mode_temperature(
            inlet_temp_k=453.0,
            outlet_temp_k=400.0,  # Large drop - subcooled
            inlet_pressure_mpa=1.0,
            outlet_pressure_mpa=0.1
        )
        assert mode == TrapFailureMode.FAILED_CLOSED

    def test_classify_normal_temperature(self):
        """Test classification of normal trap from temperature."""
        mode, confidence = classify_failure_mode_temperature(
            inlet_temp_k=453.0,
            outlet_temp_k=440.0,  # Normal subcooling
            inlet_pressure_mpa=1.0,
            outlet_pressure_mpa=0.1
        )
        assert mode == TrapFailureMode.NORMAL

    def test_combined_diagnosis_agreement(self, healthy_trap):
        """Test combined diagnosis when methods agree."""
        mode, confidence, method = classify_failure_mode(healthy_trap)

        assert mode == TrapFailureMode.NORMAL
        assert method == DiagnosticMethod.COMBINED
        assert confidence > 0.8


class TestSteamLossEstimation:
    """Test steam loss estimation."""

    def test_no_loss_normal_trap(self):
        """Test no loss for normal trap."""
        loss = estimate_steam_loss(TrapFailureMode.NORMAL, 1.0, 10.0)
        assert loss == 0.0

    def test_no_loss_failed_closed(self):
        """Test no steam loss for failed closed trap."""
        loss = estimate_steam_loss(TrapFailureMode.FAILED_CLOSED, 1.0, 10.0)
        assert loss == 0.0

    def test_loss_failed_open(self):
        """Test steam loss for failed open trap."""
        loss = estimate_steam_loss(TrapFailureMode.FAILED_OPEN, 1.0, 10.0)
        assert loss > 0

    def test_loss_increases_with_orifice_size(self):
        """Test loss increases with larger orifice."""
        loss_small = estimate_steam_loss(TrapFailureMode.FAILED_OPEN, 1.0, 5.0)
        loss_large = estimate_steam_loss(TrapFailureMode.FAILED_OPEN, 1.0, 15.0)

        assert loss_large > loss_small

    def test_loss_increases_with_pressure(self):
        """Test loss increases with higher pressure."""
        loss_low = estimate_steam_loss(TrapFailureMode.FAILED_OPEN, 0.5, 10.0)
        loss_high = estimate_steam_loss(TrapFailureMode.FAILED_OPEN, 2.0, 10.0)

        assert loss_high > loss_low

    def test_blow_through_less_than_failed_open(self):
        """Test blow-through loss is less than failed open."""
        loss_open = estimate_steam_loss(TrapFailureMode.FAILED_OPEN, 1.0, 10.0)
        loss_blow = estimate_steam_loss(TrapFailureMode.BLOW_THROUGH, 1.0, 10.0)

        assert loss_blow < loss_open


class TestEnergyLossEstimation:
    """Test energy loss estimation."""

    def test_energy_loss_positive(self):
        """Test energy loss is positive for steam loss."""
        steam_loss = 0.01  # kg/s
        energy_loss = estimate_energy_loss(steam_loss, 1.0)

        assert energy_loss > 0

    def test_energy_loss_zero_no_steam_loss(self):
        """Test no energy loss when no steam loss."""
        energy_loss = estimate_energy_loss(0.0, 1.0)
        assert energy_loss == 0.0

    def test_energy_loss_scales_with_steam_loss(self):
        """Test energy loss scales linearly with steam loss."""
        loss_1 = estimate_energy_loss(0.01, 1.0)
        loss_2 = estimate_energy_loss(0.02, 1.0)

        assert pytest.approx(loss_2, rel=0.01) == 2 * loss_1


class TestAnnualCostEstimation:
    """Test annual cost estimation."""

    def test_annual_cost_positive(self):
        """Test annual cost is positive for steam loss."""
        cost = estimate_annual_cost(0.01)
        assert cost > 0

    def test_annual_cost_zero_no_loss(self):
        """Test zero cost when no loss."""
        cost = estimate_annual_cost(0.0)
        assert cost == 0.0

    def test_annual_cost_scales_with_hours(self):
        """Test annual cost scales with operating hours."""
        cost_4000 = estimate_annual_cost(0.01, operating_hours=4000)
        cost_8000 = estimate_annual_cost(0.01, operating_hours=8000)

        assert pytest.approx(cost_8000, rel=0.01) == 2 * cost_4000


class TestMaintenancePrioritization:
    """Test maintenance priority determination."""

    def test_normal_trap_no_priority(self):
        """Test normal trap gets no maintenance priority."""
        priority = determine_maintenance_priority(
            TrapFailureMode.NORMAL, 0.0, 0.0, 0.9
        )
        assert priority == MaintenancePriority.NONE

    def test_failed_open_high_loss_critical(self):
        """Test failed open with high loss is critical."""
        priority = determine_maintenance_priority(
            TrapFailureMode.FAILED_OPEN, 100.0, 50000.0, 0.9
        )
        assert priority == MaintenancePriority.CRITICAL

    def test_failed_open_low_loss_medium(self):
        """Test failed open with low loss is medium priority."""
        priority = determine_maintenance_priority(
            TrapFailureMode.FAILED_OPEN, 10.0, 500.0, 0.9
        )
        assert priority == MaintenancePriority.MEDIUM

    def test_failed_closed_high_priority(self):
        """Test failed closed gets high priority (water hammer risk)."""
        priority = determine_maintenance_priority(
            TrapFailureMode.FAILED_CLOSED, 0.0, 0.0, 0.9
        )
        assert priority == MaintenancePriority.HIGH

    def test_unknown_low_confidence(self):
        """Test unknown with low confidence gets low priority."""
        priority = determine_maintenance_priority(
            TrapFailureMode.UNKNOWN, 0.0, 0.0, 0.3
        )
        assert priority == MaintenancePriority.LOW


class TestDiagnosticResult:
    """Test complete diagnostic analysis."""

    def test_diagnose_healthy_trap(self, healthy_trap):
        """Test diagnosis of healthy trap."""
        result = diagnose_trap(healthy_trap)

        assert result.failure_mode == TrapFailureMode.NORMAL
        assert result.steam_loss_kg_s == 0.0
        assert result.maintenance_priority == MaintenancePriority.NONE
        assert len(result.recommendations) > 0

    def test_diagnose_failed_open_trap(self, failed_open_trap):
        """Test diagnosis of failed open trap."""
        result = diagnose_trap(failed_open_trap)

        assert result.failure_mode == TrapFailureMode.FAILED_OPEN
        assert result.steam_loss_kg_s > 0
        assert result.energy_loss_kw > 0
        assert result.maintenance_priority in [MaintenancePriority.CRITICAL, MaintenancePriority.HIGH, MaintenancePriority.MEDIUM]

    def test_diagnose_failed_closed_trap(self, failed_closed_trap):
        """Test diagnosis of failed closed trap."""
        result = diagnose_trap(failed_closed_trap)

        assert result.failure_mode == TrapFailureMode.FAILED_CLOSED
        assert result.steam_loss_kg_s == 0.0
        assert result.maintenance_priority == MaintenancePriority.HIGH

    def test_diagnostic_provenance_hash(self, healthy_trap):
        """Test provenance hash is generated."""
        result = diagnose_trap(healthy_trap)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_diagnostic_recommendations_generated(self, failed_open_trap):
        """Test recommendations are generated for failed trap."""
        result = diagnose_trap(failed_open_trap)

        assert len(result.recommendations) > 0
        assert any("replace" in r.lower() for r in result.recommendations)


class TestFleetDiagnostics:
    """Test fleet-level diagnostics."""

    def test_fleet_diagnostics_structure(self, trap_fleet):
        """Test fleet diagnostics returns complete structure."""
        result = diagnose_fleet(trap_fleet)

        assert result.total_traps == len(trap_fleet)
        assert result.healthy_traps >= 0
        assert result.failed_traps >= 0
        assert result.healthy_traps + result.failed_traps <= result.total_traps

    def test_fleet_failure_rate(self, trap_fleet):
        """Test fleet failure rate calculation."""
        result = diagnose_fleet(trap_fleet)

        expected_rate = result.failed_traps / result.total_traps * 100
        assert pytest.approx(result.failure_rate_percent, rel=0.01) == expected_rate

    def test_fleet_total_losses(self, trap_fleet):
        """Test fleet total losses are summed correctly."""
        result = diagnose_fleet(trap_fleet)

        # Should have some losses from failed traps
        assert result.total_steam_loss_kg_s >= 0
        assert result.total_energy_loss_kw >= 0
        assert result.total_annual_cost >= 0

    def test_fleet_priority_breakdown(self, trap_fleet):
        """Test priority breakdown sums to total."""
        result = diagnose_fleet(trap_fleet)

        total_in_breakdown = sum(result.priority_breakdown.values())
        assert total_in_breakdown == result.total_traps

    def test_fleet_failure_mode_breakdown(self, trap_fleet):
        """Test failure mode breakdown sums to total."""
        result = diagnose_fleet(trap_fleet)

        total_in_breakdown = sum(result.failure_mode_breakdown.values())
        assert total_in_breakdown == result.total_traps

    def test_fleet_top_priority_traps(self, trap_fleet):
        """Test top priority traps are identified."""
        result = diagnose_fleet(trap_fleet)

        # Should identify failed traps as priority
        assert "TRAP-F1" in result.top_priority_traps or "TRAP-F2" in result.top_priority_traps

    def test_empty_fleet_raises_error(self):
        """Test empty fleet raises error."""
        with pytest.raises(TrapDiagnosticError):
            diagnose_fleet([])


class TestRecommendationGeneration:
    """Test recommendation generation."""

    def test_recommendations_for_normal(self, healthy_trap):
        """Test recommendations for normal trap."""
        recommendations = generate_recommendations(
            healthy_trap, TrapFailureMode.NORMAL, 0.9
        )
        assert len(recommendations) > 0
        assert any("routine" in r.lower() or "no" in r.lower() for r in recommendations)

    def test_recommendations_for_failed_open(self, failed_open_trap):
        """Test recommendations for failed open trap."""
        recommendations = generate_recommendations(
            failed_open_trap, TrapFailureMode.FAILED_OPEN, 0.9
        )
        assert any("replace" in r.lower() for r in recommendations)

    def test_recommendations_for_failed_closed(self, failed_closed_trap):
        """Test recommendations for failed closed trap."""
        recommendations = generate_recommendations(
            failed_closed_trap, TrapFailureMode.FAILED_CLOSED, 0.9
        )
        assert any("water hammer" in r.lower() for r in recommendations)

    def test_recommendations_include_trap_type_advice(self, failed_open_trap):
        """Test recommendations include trap-type-specific advice."""
        recommendations = generate_recommendations(
            failed_open_trap, TrapFailureMode.FAILED_OPEN, 0.9
        )
        # Should have trap-type specific recommendation
        assert len(recommendations) >= 2


class TestProvenanceTracking:
    """Test provenance hash generation."""

    def test_diagnostic_provenance_deterministic(self, healthy_trap):
        """Test diagnostic provenance is deterministic."""
        result1 = diagnose_trap(healthy_trap)
        result2 = diagnose_trap(healthy_trap)

        assert result1.provenance_hash == result2.provenance_hash

    def test_fleet_provenance_deterministic(self, trap_fleet):
        """Test fleet provenance is deterministic."""
        result1 = diagnose_fleet(trap_fleet)
        result2 = diagnose_fleet(trap_fleet)

        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_changes_with_input(self):
        """Test provenance changes when input changes."""
        trap1 = TrapData("T1", TrapType.THERMODYNAMIC, "A", 1.0, 0.1, 453, 440, 65, None, 10, 10)
        trap2 = TrapData("T2", TrapType.THERMODYNAMIC, "A", 1.0, 0.1, 453, 440, 65, None, 10, 10)

        result1 = diagnose_trap(trap1)
        result2 = diagnose_trap(trap2)

        assert result1.provenance_hash != result2.provenance_hash
