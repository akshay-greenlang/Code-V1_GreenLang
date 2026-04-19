"""
GL-016 WATERGUARD Boiler Water Treatment Agent - Coordinators Module

This module provides coordinators for managing chemistry calculations,
optimization routines, and safety gate checks. Coordinators orchestrate
the agent's core functionality with zero-hallucination deterministic
calculations and full provenance tracking.

Standards Compliance:
    - ASME Boiler and Pressure Vessel Code
    - ABMA (American Boiler Manufacturers Association) Guidelines
    - IEC 62443 (Industrial Cybersecurity)
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import asyncio
import hashlib
import logging

from .config import (
    WaterguardConfig,
    ChemistryLimitsConfig,
    BlowdownConfig,
    DosingConfig,
    SafetyConfig,
    ConstraintType,
    ComplianceStatus,
    ChemicalType,
    OperatingMode,
)
from .schemas import (
    WaterChemistryInput,
    FeedwaterChemistryInput,
    BoilerOperatingInput,
    CyclesOfConcentrationResult,
    CoCCalculationMethod,
    BlowdownRecommendation,
    DosingRecommendation,
    DosingMode,
    ComplianceViolation,
    ComplianceWarning,
    ConstraintDistance,
    ComplianceStatusResult,
    ChemistryState,
    CalculationStep,
    ProvenanceRecord,
    ReasonCode,
    SeverityLevel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BASE COORDINATOR
# =============================================================================

class BaseCoordinator(ABC):
    """Base class for all coordinators."""
    
    def __init__(self, config: WaterguardConfig) -> None:
        self.config = config
        self._initialized = False
        self._last_run: Optional[datetime] = None
        self._run_count = 0
        self._error_count = 0
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the coordinator."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the coordinator."""
        pass
    
    def _compute_hash(self, data: str) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _create_provenance(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        steps: List[CalculationStep],
    ) -> ProvenanceRecord:
        """Create a provenance record for audit trail."""
        input_str = str(sorted(inputs.items()))
        output_str = str(sorted(outputs.items()))
        
        input_hash = self._compute_hash(input_str)
        output_hash = self._compute_hash(output_str)
        provenance_hash = self._compute_hash(f"{input_hash}{output_hash}")
        
        return ProvenanceRecord(
            input_hash=input_hash,
            output_hash=output_hash,
            provenance_hash=provenance_hash,
            steps=steps,
            deterministic=True,
        )


# =============================================================================
# CHEMISTRY COORDINATOR
# =============================================================================

class ChemistryCoordinator(BaseCoordinator):
    """
    Coordinator for chemistry calculations and constraint checking.
    
    Manages:
        - Cycles of Concentration (CoC) calculation
        - Chemistry limit compliance checking
        - Distance-to-limit calculations
        - Trend analysis for predictive warnings
    
    All calculations are deterministic (zero-hallucination) with
    full provenance tracking via SHA-256 hashes.
    """
    
    def __init__(self, config: WaterguardConfig) -> None:
        super().__init__(config)
        self._chemistry_limits = config.chemistry_limits
        self._last_chemistry_state: Optional[ChemistryState] = None
        self._chemistry_history: List[Dict[str, float]] = []
    
    async def initialize(self) -> None:
        """Initialize the chemistry coordinator."""
        logger.info(f"Initializing ChemistryCoordinator for {self.config.agent_id}")
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Shutdown the chemistry coordinator."""
        logger.info(f"Shutting down ChemistryCoordinator for {self.config.agent_id}")
        self._initialized = False
    
    def calculate_cycles_of_concentration(
        self,
        boiler_water: WaterChemistryInput,
        feedwater: FeedwaterChemistryInput,
        method: CoCCalculationMethod = CoCCalculationMethod.CONDUCTIVITY_RATIO,
    ) -> CyclesOfConcentrationResult:
        """
        Calculate Cycles of Concentration (CoC).
        
        CoC = Boiler Water Concentration / Feedwater Concentration
        
        This is a DETERMINISTIC calculation - no LLM involvement.
        
        Args:
            boiler_water: Boiler water chemistry input
            feedwater: Feedwater chemistry input
            method: Calculation method to use
        
        Returns:
            CyclesOfConcentrationResult with CoC value and provenance hash
        """
        self._run_count += 1
        self._last_run = datetime.now(timezone.utc)
        
        steps: List[CalculationStep] = []
        
        # Select calculation method
        if method == CoCCalculationMethod.CONDUCTIVITY_RATIO:
            if feedwater.conductivity_us_cm <= 0:
                raise ValueError("Feedwater conductivity must be greater than 0")
            
            coc_value = boiler_water.conductivity_us_cm / feedwater.conductivity_us_cm
            
            steps.append(CalculationStep(
                step_number=1,
                description="Calculate CoC using conductivity ratio",
                formula="CoC = Boiler_Conductivity / Feedwater_Conductivity",
                inputs={
                    "boiler_conductivity_us_cm": boiler_water.conductivity_us_cm,
                    "feedwater_conductivity_us_cm": feedwater.conductivity_us_cm,
                },
                output=coc_value,
                unit="dimensionless",
            ))
            
            fw_cond = feedwater.conductivity_us_cm
            bw_cond = boiler_water.conductivity_us_cm
        
        elif method == CoCCalculationMethod.SILICA_RATIO:
            if feedwater.silica_ppm is None or feedwater.silica_ppm <= 0:
                raise ValueError("Feedwater silica must be greater than 0")
            
            coc_value = boiler_water.silica_ppm / feedwater.silica_ppm
            
            steps.append(CalculationStep(
                step_number=1,
                description="Calculate CoC using silica ratio",
                formula="CoC = Boiler_Silica / Feedwater_Silica",
                inputs={
                    "boiler_silica_ppm": boiler_water.silica_ppm,
                    "feedwater_silica_ppm": feedwater.silica_ppm,
                },
                output=coc_value,
                unit="dimensionless",
            ))
            
            fw_cond = feedwater.conductivity_us_cm
            bw_cond = boiler_water.conductivity_us_cm
        
        else:
            # Default to conductivity ratio
            if feedwater.conductivity_us_cm <= 0:
                raise ValueError("Feedwater conductivity must be greater than 0")
            coc_value = boiler_water.conductivity_us_cm / feedwater.conductivity_us_cm
            fw_cond = feedwater.conductivity_us_cm
            bw_cond = boiler_water.conductivity_us_cm
        
        # Compute provenance hash
        hash_input = f"{boiler_water.conductivity_us_cm}{feedwater.conductivity_us_cm}{coc_value}{method.value}"
        computation_hash = self._compute_hash(hash_input)
        
        # Determine confidence based on data quality
        confidence = 95.0
        if boiler_water.quality_flag.value != "good":
            confidence -= 10.0
        if feedwater.quality_flag.value != "good":
            confidence -= 10.0
        
        result = CyclesOfConcentrationResult(
            coc_value=round(coc_value, 2),
            method=method,
            computation_hash=computation_hash,
            feedwater_conductivity_us_cm=fw_cond,
            boiler_conductivity_us_cm=bw_cond,
            confidence_percent=max(confidence, 50.0),
        )
        
        logger.debug(f"CoC calculated: {result.coc_value} using {method.value}")
        return result

    
    def check_compliance(
        self,
        boiler_water: WaterChemistryInput,
    ) -> ComplianceStatusResult:
        """
        Check chemistry compliance against configured limits.
        
        This is a DETERMINISTIC check - no LLM involvement.
        
        Args:
            boiler_water: Current boiler water chemistry
        
        Returns:
            ComplianceStatusResult with violations, warnings, and distances
        """
        violations: List[ComplianceViolation] = []
        warnings: List[ComplianceWarning] = []
        distances: List[ConstraintDistance] = []
        
        limits = self._chemistry_limits
        
        # Check conductivity
        cond = boiler_water.conductivity_us_cm
        cond_max = limits.conductivity.max_us_cm
        cond_hard = limits.conductivity.hard_limit_us_cm
        
        if cond > cond_hard:
            violations.append(ComplianceViolation(
                parameter="conductivity",
                current_value=cond,
                limit_value=cond_hard,
                severity=SeverityLevel.CRITICAL,
                constraint_type=ConstraintType.HARD,
                reason_code=ReasonCode.HIGH_CONDUCTIVITY,
            ))
        elif cond > cond_max:
            violations.append(ComplianceViolation(
                parameter="conductivity",
                current_value=cond,
                limit_value=cond_max,
                severity=SeverityLevel.WARNING,
                constraint_type=ConstraintType.SOFT,
                reason_code=ReasonCode.HIGH_CONDUCTIVITY,
            ))
        elif cond > cond_max * 0.9:
            warnings.append(ComplianceWarning(
                parameter="conductivity",
                current_value=cond,
                limit_value=cond_max,
                distance_percent=((cond_max - cond) / cond_max) * 100,
                reason_code=ReasonCode.HIGH_CONDUCTIVITY,
            ))
        
        distances.append(ConstraintDistance(
            parameter="conductivity",
            current_value=cond,
            limit_value=cond_max,
            distance_percent=((cond_max - cond) / cond_max) * 100 if cond < cond_max else 0,
            constraint_type=ConstraintType.HARD,
        ))
        
        # Check silica
        silica = boiler_water.silica_ppm
        silica_max = limits.silica.max_ppm
        silica_hard = limits.silica.hard_limit_ppm
        
        if silica > silica_hard:
            violations.append(ComplianceViolation(
                parameter="silica",
                current_value=silica,
                limit_value=silica_hard,
                severity=SeverityLevel.CRITICAL,
                constraint_type=ConstraintType.HARD,
                reason_code=ReasonCode.HIGH_SILICA,
            ))
        elif silica > silica_max:
            violations.append(ComplianceViolation(
                parameter="silica",
                current_value=silica,
                limit_value=silica_max,
                severity=SeverityLevel.WARNING,
                constraint_type=ConstraintType.SOFT,
                reason_code=ReasonCode.HIGH_SILICA,
            ))
        
        distances.append(ConstraintDistance(
            parameter="silica",
            current_value=silica,
            limit_value=silica_max,
            distance_percent=((silica_max - silica) / silica_max) * 100 if silica < silica_max else 0,
            constraint_type=ConstraintType.HARD,
        ))
        
        # Check pH
        ph = boiler_water.ph
        ph_min = limits.ph.min_ph
        ph_max = limits.ph.max_ph
        
        if ph < ph_min:
            violations.append(ComplianceViolation(
                parameter="ph",
                current_value=ph,
                limit_value=ph_min,
                severity=SeverityLevel.CRITICAL,
                constraint_type=ConstraintType.HARD,
                reason_code=ReasonCode.LOW_PH,
            ))
        elif ph > ph_max:
            violations.append(ComplianceViolation(
                parameter="ph",
                current_value=ph,
                limit_value=ph_max,
                severity=SeverityLevel.CRITICAL,
                constraint_type=ConstraintType.HARD,
                reason_code=ReasonCode.HIGH_PH,
            ))
        
        # Determine overall status
        if any(v.severity == SeverityLevel.CRITICAL for v in violations):
            overall_status = ComplianceStatus.CRITICAL
        elif violations:
            overall_status = ComplianceStatus.VIOLATION
        elif warnings:
            overall_status = ComplianceStatus.WARNING
        else:
            overall_status = ComplianceStatus.COMPLIANT
        
        # Compute provenance hash
        hash_input = f"{cond}{silica}{ph}{len(violations)}{len(warnings)}"
        computation_hash = self._compute_hash(hash_input)
        
        return ComplianceStatusResult(
            all_constraints_met=len(violations) == 0,
            overall_status=overall_status,
            violations=violations,
            warnings=warnings,
            distances_to_limits=distances,
            computation_hash=computation_hash,
        )


# =============================================================================
# OPTIMIZATION COORDINATOR
# =============================================================================

class OptimizationCoordinator(BaseCoordinator):
    """
    Coordinator for blowdown and dosing optimization.
    
    Manages:
        - Blowdown rate optimization to maintain target CoC
        - Chemical dosing optimization for corrosion/scale control
        - Energy-aware optimization considering heat recovery
    
    All optimizations are deterministic (zero-hallucination) using
    rule-based control with configurable setpoints.
    """
    
    def __init__(self, config: WaterguardConfig) -> None:
        super().__init__(config)
        self._blowdown_config = config.blowdown
        self._phosphate_dosing = config.phosphate_dosing
        self._o2_scavenger_dosing = config.oxygen_scavenger_dosing
        self._current_blowdown_setpoint: float = config.blowdown.continuous_target_percent
    
    async def initialize(self) -> None:
        logger.info(f"Initializing OptimizationCoordinator for {self.config.agent_id}")
        self._initialized = True
    
    async def shutdown(self) -> None:
        logger.info(f"Shutting down OptimizationCoordinator for {self.config.agent_id}")
        self._initialized = False
    
    def optimize_blowdown(
        self,
        current_coc: CyclesOfConcentrationResult,
        compliance: ComplianceStatusResult,
        boiler_data: BoilerOperatingInput,
    ) -> BlowdownRecommendation:
        """
        Calculate optimal blowdown rate.
        
        This is a DETERMINISTIC calculation using rule-based control.
        
        Args:
            current_coc: Current cycles of concentration
            compliance: Current compliance status
            boiler_data: Current boiler operating data
        
        Returns:
            BlowdownRecommendation with target setpoint
        """
        self._run_count += 1
        self._last_run = datetime.now(timezone.utc)
        
        target_coc = self._blowdown_config.coc_target
        coc_min = self._blowdown_config.coc_min
        coc_max = self._blowdown_config.coc_max
        
        current_coc_value = current_coc.coc_value
        current_blowdown = boiler_data.blowdown_flow_kg_s or 0
        steam_flow = boiler_data.steam_flow_kg_s
        
        # Calculate current blowdown percentage
        if steam_flow > 0:
            current_percent = (current_blowdown / steam_flow) * 100
        else:
            current_percent = self._current_blowdown_setpoint
        
        # Determine reason code and adjustment
        if current_coc_value > coc_max:
            # CoC too high - increase blowdown
            reason_code = ReasonCode.COC_TOO_HIGH
            adjustment_factor = 1.0 + (current_coc_value - coc_max) / coc_max * 0.5
            new_setpoint = min(current_percent * adjustment_factor, self._blowdown_config.continuous_max_percent)
        elif current_coc_value < coc_min:
            # CoC too low - decrease blowdown
            reason_code = ReasonCode.COC_TOO_LOW
            adjustment_factor = 1.0 - (coc_min - current_coc_value) / coc_min * 0.3
            new_setpoint = max(current_percent * adjustment_factor, self._blowdown_config.continuous_min_percent)
        elif not compliance.all_constraints_met:
            # Compliance violation - increase blowdown
            reason_code = ReasonCode.HIGH_CONDUCTIVITY if any(
                v.parameter == "conductivity" for v in compliance.violations
            ) else ReasonCode.OPTIMIZE_EFFICIENCY
            new_setpoint = min(current_percent * 1.2, self._blowdown_config.continuous_max_percent)
        else:
            # Normal operation - maintain setpoint
            reason_code = ReasonCode.OPTIMIZE_EFFICIENCY
            new_setpoint = self._blowdown_config.continuous_target_percent
        
        # Round to reasonable precision
        new_setpoint = round(new_setpoint, 1)
        
        # Calculate expected CoC after adjustment
        if new_setpoint > 0:
            expected_coc = current_coc_value * (current_percent / new_setpoint) if current_percent > 0 else target_coc
            expected_coc = min(max(expected_coc, coc_min), coc_max)
        else:
            expected_coc = target_coc
        
        # Calculate energy impact (approximate)
        energy_impact_kw = None
        if steam_flow > 0:
            delta_blowdown = (new_setpoint - current_percent) / 100 * steam_flow
            # Approximate energy loss: 1 kg/s blowdown at 10 bar ~ 250 kW
            energy_impact_kw = delta_blowdown * 250
        
        self._current_blowdown_setpoint = new_setpoint
        
        return BlowdownRecommendation(
            target_setpoint_percent=new_setpoint,
            current_value_percent=round(current_percent, 1),
            reason_code=reason_code,
            constraint_distances=compliance.distances_to_limits,
            time_to_violation_minutes=None,
            expected_coc_after=round(expected_coc, 2),
            energy_impact_kw=round(energy_impact_kw, 1) if energy_impact_kw else None,
            confidence_percent=current_coc.confidence_percent,
        )

    
    def optimize_dosing(
        self,
        boiler_water: WaterChemistryInput,
        compliance: ComplianceStatusResult,
    ) -> List[DosingRecommendation]:
        """
        Calculate optimal chemical dosing rates.
        
        This is a DETERMINISTIC calculation using rule-based control.
        
        Args:
            boiler_water: Current boiler water chemistry
            compliance: Current compliance status
        
        Returns:
            List of DosingRecommendation for each chemical
        """
        recommendations: List[DosingRecommendation] = []
        limits = self.config.chemistry_limits
        
        # Phosphate dosing
        if self._phosphate_dosing.enabled:
            phosphate_current = boiler_water.phosphate_ppm or 0
            phosphate_target = (limits.phosphate_min_ppm + limits.phosphate_max_ppm) / 2
            
            if phosphate_current < limits.phosphate_min_ppm:
                # Need to increase phosphate
                deficit = phosphate_target - phosphate_current
                rate = min(
                    deficit * 2.0,  # 2 ml/min per ppm deficit
                    self._phosphate_dosing.pump_capacity_ml_min * self._phosphate_dosing.max_dose_percent / 100
                )
                recommendations.append(DosingRecommendation(
                    chemical_type=ChemicalType.PHOSPHATE,
                    rate_ml_min=round(rate, 1),
                    mode=DosingMode.CONTINUOUS,
                    reason_code=ReasonCode.LOW_ALKALINITY,
                    target_residual_ppm=phosphate_target,
                    current_residual_ppm=phosphate_current,
                    confidence_percent=90.0,
                ))
            elif phosphate_current > limits.phosphate_max_ppm:
                # Reduce or stop dosing
                recommendations.append(DosingRecommendation(
                    chemical_type=ChemicalType.PHOSPHATE,
                    rate_ml_min=0.0,
                    mode=DosingMode.CONTINUOUS,
                    reason_code=ReasonCode.HIGH_ALKALINITY,
                    target_residual_ppm=phosphate_target,
                    current_residual_ppm=phosphate_current,
                    confidence_percent=95.0,
                ))
        
        # Oxygen scavenger dosing
        if self._o2_scavenger_dosing.enabled:
            sulfite_current = boiler_water.sulfite_ppm or 0
            sulfite_target = (limits.sulfite_min_ppm + limits.sulfite_max_ppm) / 2
            do2 = boiler_water.dissolved_o2_ppb or 0
            
            if sulfite_current < limits.sulfite_min_ppm or do2 > limits.dissolved_o2.max_ppb:
                deficit = sulfite_target - sulfite_current
                rate = min(
                    max(deficit * 1.5, do2 * 0.1),
                    self._o2_scavenger_dosing.pump_capacity_ml_min * self._o2_scavenger_dosing.max_dose_percent / 100
                )
                recommendations.append(DosingRecommendation(
                    chemical_type=ChemicalType.OXYGEN_SCAVENGER,
                    rate_ml_min=round(rate, 1),
                    mode=DosingMode.CONTINUOUS,
                    reason_code=ReasonCode.HIGH_DISSOLVED_O2,
                    target_residual_ppm=sulfite_target,
                    current_residual_ppm=sulfite_current,
                    confidence_percent=90.0,
                ))
        
        return recommendations


# =============================================================================
# SAFETY COORDINATOR
# =============================================================================

class SafetyCoordinator(BaseCoordinator):
    """
    Coordinator for safety gates and interlocks.
    
    Manages:
        - Safety gate checks before control actions
        - Interlock activation on limit violations
        - Failsafe mode activation
        - Safety event generation
    
    All safety checks are deterministic and follow IEC 61511 principles.
    """
    
    def __init__(self, config: WaterguardConfig) -> None:
        super().__init__(config)
        self._safety_config = config.safety
        self._active_interlocks: Dict[str, Dict[str, Any]] = {}
        self._failsafe_active = False
        self._last_heartbeat: Optional[datetime] = None
    
    async def initialize(self) -> None:
        logger.info(f"Initializing SafetyCoordinator for {self.config.agent_id}")
        self._initialized = True
        self._last_heartbeat = datetime.now(timezone.utc)
    
    async def shutdown(self) -> None:
        logger.info(f"Shutting down SafetyCoordinator for {self.config.agent_id}")
        self._initialized = False
    
    def check_safety_gates(
        self,
        boiler_water: WaterChemistryInput,
        compliance: ComplianceStatusResult,
    ) -> Tuple[bool, List[str]]:
        """
        Check all safety gates before allowing control actions.
        
        Args:
            boiler_water: Current boiler water chemistry
            compliance: Current compliance status
        
        Returns:
            Tuple of (all_gates_passed, list_of_failed_gates)
        """
        failed_gates: List[str] = []
        limits = self.config.chemistry_limits
        
        # Check conductivity interlock
        if self._safety_config.high_conductivity_interlock:
            if boiler_water.conductivity_us_cm > limits.conductivity.hard_limit_us_cm:
                failed_gates.append("HIGH_CONDUCTIVITY_INTERLOCK")
                self._activate_interlock("HIGH_CONDUCTIVITY_INTERLOCK", boiler_water.conductivity_us_cm)
        
        # Check silica interlock
        if self._safety_config.high_silica_interlock:
            if boiler_water.silica_ppm > limits.silica.hard_limit_ppm:
                failed_gates.append("HIGH_SILICA_INTERLOCK")
                self._activate_interlock("HIGH_SILICA_INTERLOCK", boiler_water.silica_ppm)
        
        # Check pH interlock
        if self._safety_config.ph_interlock:
            if boiler_water.ph < limits.ph.min_ph or boiler_water.ph > limits.ph.max_ph:
                failed_gates.append("PH_OUT_OF_RANGE_INTERLOCK")
                self._activate_interlock("PH_OUT_OF_RANGE_INTERLOCK", boiler_water.ph)
        
        # Check for critical violations
        if compliance.overall_status == ComplianceStatus.CRITICAL:
            failed_gates.append("CRITICAL_VIOLATION_INTERLOCK")
        
        all_passed = len(failed_gates) == 0
        
        if not all_passed:
            logger.warning(f"Safety gates failed: {failed_gates}")
        
        return all_passed, failed_gates
    
    def _activate_interlock(self, interlock_name: str, trigger_value: float) -> None:
        """Activate an interlock."""
        if interlock_name not in self._active_interlocks:
            self._active_interlocks[interlock_name] = {
                "activated_at": datetime.now(timezone.utc),
                "trigger_value": trigger_value,
            }
            logger.critical(f"INTERLOCK ACTIVATED: {interlock_name} (value: {trigger_value})")
    
    def clear_interlock(self, interlock_name: str, user: str) -> bool:
        """Clear an interlock (requires operator action)."""
        if interlock_name in self._active_interlocks:
            del self._active_interlocks[interlock_name]
            logger.info(f"Interlock {interlock_name} cleared by {user}")
            return True
        return False
    
    def get_active_interlocks(self) -> Dict[str, Dict[str, Any]]:
        """Get all active interlocks."""
        return dict(self._active_interlocks)
    
    def is_action_permitted(
        self,
        action_type: str,
        operating_mode: OperatingMode,
    ) -> Tuple[bool, str]:
        """
        Check if a control action is permitted.
        
        Args:
            action_type: Type of action (e.g., "blowdown_change", "dosing_start")
            operating_mode: Current operating mode
        
        Returns:
            Tuple of (permitted, reason)
        """
        # Check for active interlocks
        if self._active_interlocks:
            return False, f"Active interlocks: {list(self._active_interlocks.keys())}"
        
        # Check failsafe mode
        if self._failsafe_active:
            if action_type == "dosing_start" and not self._safety_config.failsafe_dosing_enabled:
                return False, "Dosing not permitted in failsafe mode"
        
        # Check operating mode
        if operating_mode == OperatingMode.RECOMMEND_ONLY:
            return False, "Control actions not permitted in RECOMMEND_ONLY mode"
        
        if operating_mode == OperatingMode.FALLBACK:
            if action_type not in ["blowdown_increase", "emergency_stop"]:
                return False, "Only safety actions permitted in FALLBACK mode"
        
        return True, "Action permitted"
    
    def heartbeat(self) -> bool:
        """
        Safety heartbeat - must be called regularly.
        
        Returns:
            True if within timeout, False if watchdog should trigger
        """
        now = datetime.now(timezone.utc)
        
        if self._last_heartbeat:
            elapsed_ms = (now - self._last_heartbeat).total_seconds() * 1000
            if elapsed_ms > self._safety_config.watchdog_timeout_ms:
                logger.critical(f"Safety watchdog timeout! Elapsed: {elapsed_ms}ms")
                self._failsafe_active = True
                return False
        
        self._last_heartbeat = now
        return True
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        return {
            "initialized": self._initialized,
            "failsafe_active": self._failsafe_active,
            "active_interlocks": list(self._active_interlocks.keys()),
            "interlock_count": len(self._active_interlocks),
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            "safety_level": self._safety_config.safety_level.value,
        }
