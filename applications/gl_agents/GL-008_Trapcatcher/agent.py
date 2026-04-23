# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER Agent

Main steam trap monitoring and diagnostic agent for GreenLang platform.
Orchestrates acoustic analysis, thermal analysis, classification,
explainability, and reporting components.

Standards:
- ASME PTC 39: Steam Traps - Performance Test Codes
- DOE Steam System Assessment Protocol
- ISO 7841: Automatic steam traps - Steam loss determination

Key Capabilities:
- Real-time steam trap monitoring
- Multimodal diagnostic classification
- Energy loss quantification
- SHAP-compatible explainability
- CO2e emissions reporting
- Maintenance prioritization

Zero-Hallucination Guarantee:
All calculations use deterministic engineering formulas.
No LLM or AI inference in any calculation path.
Same inputs always produce identical outputs.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from .calculators.steam_trap_energy_loss_calculator import (
    SteamTrapEnergyLossCalculator,
    TrapEnergyInput,
)
from .calculators.trap_population_analyzer import (
    TrapPopulationAnalyzer,
    TrapRecord,
    create_trap_record,
)
from .core.trap_state_classifier import (
    TrapStateClassifier,
    SensorInput,
    ClassificationResult,
    TrapCondition,
)
from .explainability.diagnostic_explainer import (
    DiagnosticExplainer,
    ExplanationResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class AgentMode(str, Enum):
    """Agent operating modes."""
    MONITORING = "monitoring"      # Continuous monitoring mode
    SURVEY = "survey"              # Batch survey mode
    DIAGNOSTIC = "diagnostic"      # Single trap diagnostic
    REPORTING = "reporting"        # Report generation mode


class AlertLevel(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    NONE = "none"


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AgentConfig:
    """
    Configuration for TRAPCATCHER agent.

    Attributes:
        agent_id: Unique agent identifier
        agent_name: Human-readable name
        version: Agent version
        mode: Operating mode
        polling_interval_seconds: Monitoring poll interval
        alert_threshold_confidence: Minimum confidence for alerts
        energy_cost_per_mwh_usd: Energy cost for ROI calculations
        co2_factor_kg_per_mwh: CO2 emission factor
    """
    agent_id: str = "GL-008"
    agent_name: str = "TRAPCATCHER"
    version: str = "1.0.0"
    mode: AgentMode = AgentMode.DIAGNOSTIC
    polling_interval_seconds: float = 60.0
    alert_threshold_confidence: float = 0.70
    energy_cost_per_mwh_usd: float = 35.0
    co2_factor_kg_per_mwh: float = 180.0  # Natural gas boiler


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TrapDiagnosticInput:
    """
    Input data for single trap diagnostic.

    Attributes:
        trap_id: Unique trap identifier
        acoustic_amplitude_db: Peak acoustic amplitude (dB)
        acoustic_frequency_khz: Dominant frequency (kHz)
        inlet_temp_c: Inlet temperature (Celsius)
        outlet_temp_c: Outlet temperature (Celsius)
        pressure_bar_g: Operating pressure (bar gauge)
        trap_type: Type of steam trap
        orifice_diameter_mm: Orifice diameter (mm)
        trap_age_years: Age of trap in years
        last_maintenance_days: Days since last maintenance
        location: Physical location
        system: Steam system identifier
    """
    trap_id: str
    acoustic_amplitude_db: Optional[float] = None
    acoustic_frequency_khz: Optional[float] = None
    inlet_temp_c: Optional[float] = None
    outlet_temp_c: Optional[float] = None
    pressure_bar_g: float = 10.0
    trap_type: str = "thermodynamic"
    orifice_diameter_mm: float = 6.35
    trap_age_years: float = 0.0
    last_maintenance_days: int = 0
    location: str = ""
    system: str = ""


@dataclass
class DiagnosticOutput:
    """
    Complete diagnostic output for a trap.

    Attributes:
        trap_id: Steam trap identifier
        timestamp: Diagnostic timestamp
        condition: Classified condition
        severity: Severity level
        confidence: Classification confidence
        energy_loss_kw: Estimated energy loss (kW)
        annual_cost_usd: Annual cost of energy loss
        annual_co2_kg: Annual CO2 emissions
        classification_result: Full classification result
        explanation: SHAP-compatible explanation
        recommended_action: Recommended action
        alert_level: Alert level
        provenance_hash: SHA-256 hash for audit trail
    """
    trap_id: str
    timestamp: datetime
    condition: str
    severity: str
    confidence: float
    energy_loss_kw: float
    annual_cost_usd: float
    annual_co2_kg: float
    classification_result: ClassificationResult
    explanation: Optional[ExplanationResult]
    recommended_action: str
    alert_level: AlertLevel
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trap_id": self.trap_id,
            "timestamp": self.timestamp.isoformat(),
            "condition": self.condition,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "energy_loss_kw": round(self.energy_loss_kw, 2),
            "annual_cost_usd": round(self.annual_cost_usd, 2),
            "annual_co2_kg": round(self.annual_co2_kg, 2),
            "recommended_action": self.recommended_action,
            "alert_level": self.alert_level.value,
            "classification": self.classification_result.to_dict(),
            "provenance_hash": self.provenance_hash
        }


@dataclass
class FleetSummary:
    """
    Fleet-wide summary statistics.

    Attributes:
        total_traps: Total number of traps analyzed
        healthy_count: Number of healthy traps
        failed_count: Number of failed traps
        leaking_count: Number of leaking traps
        unknown_count: Number with unknown status
        total_energy_loss_kw: Total energy loss across fleet
        total_annual_cost_usd: Total annual cost
        total_annual_co2_kg: Total annual CO2 emissions
        fleet_health_score: Overall fleet health (0-100)
        critical_alerts: Number of critical alerts
    """
    total_traps: int
    healthy_count: int
    failed_count: int
    leaking_count: int
    unknown_count: int
    total_energy_loss_kw: float
    total_annual_cost_usd: float
    total_annual_co2_kg: float
    fleet_health_score: float
    critical_alerts: int


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class TrapcatcherAgent:
    """
    GL-008 TRAPCATCHER Steam Trap Monitoring Agent.

    Main orchestrator for steam trap monitoring, diagnostics, and reporting.
    Coordinates acoustic analysis, thermal analysis, classification,
    explainability, and energy loss calculations.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations use deterministic engineering formulas
    - No LLM or AI inference in any calculation path
    - Same inputs always produce identical outputs
    - Complete provenance tracking with SHA-256 hashes

    Key Capabilities:
    1. Single trap diagnostics with full explainability
    2. Fleet-wide monitoring and prioritization
    3. Energy loss quantification with CO2e
    4. Maintenance prioritization

    Example:
        >>> agent = TrapcatcherAgent()
        >>> result = agent.diagnose_trap(TrapDiagnosticInput(
        ...     trap_id="ST-001",
        ...     acoustic_amplitude_db=75.0,
        ...     inlet_temp_c=185.0,
        ...     outlet_temp_c=180.0,
        ...     pressure_bar_g=10.0
        ... ))
        >>> print(f"Condition: {result.condition}")
        >>> print(f"Energy loss: {result.energy_loss_kw:.1f} kW")
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize TRAPCATCHER agent.

        Args:
            config: Agent configuration (uses defaults if not provided)
        """
        self.config = config or AgentConfig()

        # Initialize components
        self._classifier = TrapStateClassifier()
        self._explainer = DiagnosticExplainer()
        self._energy_calculator = SteamTrapEnergyLossCalculator()
        self._population_analyzer = TrapPopulationAnalyzer()

        # Statistics
        self._diagnostic_count = 0
        self._alert_count = 0

        logger.info(
            f"{self.config.agent_name} Agent v{self.config.version} initialized "
            f"(mode={self.config.mode.value})"
        )

    def diagnose_trap(
        self,
        input_data: TrapDiagnosticInput,
        include_explanation: bool = True
    ) -> DiagnosticOutput:
        """
        Perform comprehensive diagnostic on a single trap.

        ZERO-HALLUCINATION: Uses deterministic classification and calculations.

        Args:
            input_data: Trap diagnostic input data
            include_explanation: Whether to include SHAP explanation

        Returns:
            DiagnosticOutput with complete analysis
        """
        self._diagnostic_count += 1
        timestamp = datetime.now(timezone.utc)

        # Build sensor input for classifier
        sensor_input = SensorInput(
            trap_id=input_data.trap_id,
            timestamp=timestamp,
            acoustic_amplitude_db=input_data.acoustic_amplitude_db,
            acoustic_frequency_khz=input_data.acoustic_frequency_khz,
            inlet_temp_c=input_data.inlet_temp_c,
            outlet_temp_c=input_data.outlet_temp_c,
            pressure_bar_g=input_data.pressure_bar_g,
            trap_type=input_data.trap_type,
            trap_age_years=input_data.trap_age_years,
            last_maintenance_days=input_data.last_maintenance_days,
            location=input_data.location
        )

        # Classify trap condition
        classification = self._classifier.classify(sensor_input)

        # Calculate energy loss
        energy_loss_kw = self._calculate_energy_loss(
            classification.condition, input_data
        )

        # Calculate annual costs
        operating_hours = 8760  # Hours per year
        energy_loss_mwh_year = (energy_loss_kw / 1000) * operating_hours
        annual_cost_usd = energy_loss_mwh_year * self.config.energy_cost_per_mwh_usd
        annual_co2_kg = energy_loss_mwh_year * self.config.co2_factor_kg_per_mwh

        # Generate explanation if requested
        explanation = None
        if include_explanation:
            features = {
                "acoustic_amplitude_db": input_data.acoustic_amplitude_db or 0,
                "temperature_differential_c": (
                    (input_data.inlet_temp_c or 0) - (input_data.outlet_temp_c or 0)
                ),
                "trap_age_years": input_data.trap_age_years,
                "pressure_bar_g": input_data.pressure_bar_g
            }
            explanation = self._explainer.explain(
                trap_id=input_data.trap_id,
                classification=classification.condition.value,
                confidence=classification.confidence_score,
                features=features
            )

        # Determine alert level
        alert_level = self._determine_alert_level(classification)
        if alert_level in [AlertLevel.CRITICAL, AlertLevel.WARNING]:
            self._alert_count += 1

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data, classification, energy_loss_kw
        )

        return DiagnosticOutput(
            trap_id=input_data.trap_id,
            timestamp=timestamp,
            condition=classification.condition.value,
            severity=classification.severity.value,
            confidence=classification.confidence_score,
            energy_loss_kw=energy_loss_kw,
            annual_cost_usd=annual_cost_usd,
            annual_co2_kg=annual_co2_kg,
            classification_result=classification,
            explanation=explanation,
            recommended_action=classification.recommended_action,
            alert_level=alert_level,
            provenance_hash=provenance_hash
        )

    def analyze_fleet(
        self,
        traps: List[TrapDiagnosticInput]
    ) -> Tuple[List[DiagnosticOutput], FleetSummary]:
        """
        Analyze entire fleet of steam traps.

        Args:
            traps: List of trap diagnostic inputs

        Returns:
            Tuple of (list of diagnostic outputs, fleet summary)
        """
        results = []
        healthy = 0
        failed = 0
        leaking = 0
        unknown = 0
        total_energy_loss = 0.0
        critical_alerts = 0

        for trap_input in traps:
            try:
                result = self.diagnose_trap(trap_input, include_explanation=False)
                results.append(result)

                # Count by condition
                if result.condition == TrapCondition.OPERATING_NORMAL.value:
                    healthy += 1
                elif result.condition in [TrapCondition.FAILED_OPEN.value, TrapCondition.FAILED_CLOSED.value]:
                    failed += 1
                elif result.condition == TrapCondition.LEAKING.value:
                    leaking += 1
                else:
                    unknown += 1

                total_energy_loss += result.energy_loss_kw

                if result.alert_level == AlertLevel.CRITICAL:
                    critical_alerts += 1

            except Exception as e:
                logger.error(f"Error analyzing trap {trap_input.trap_id}: {e}")
                unknown += 1

        # Calculate totals
        total = len(traps)
        operating_hours = 8760
        total_mwh_year = (total_energy_loss / 1000) * operating_hours
        total_cost = total_mwh_year * self.config.energy_cost_per_mwh_usd
        total_co2 = total_mwh_year * self.config.co2_factor_kg_per_mwh

        # Fleet health score
        health_score = (healthy / total * 100) if total > 0 else 0

        summary = FleetSummary(
            total_traps=total,
            healthy_count=healthy,
            failed_count=failed,
            leaking_count=leaking,
            unknown_count=unknown,
            total_energy_loss_kw=round(total_energy_loss, 2),
            total_annual_cost_usd=round(total_cost, 2),
            total_annual_co2_kg=round(total_co2, 2),
            fleet_health_score=round(health_score, 1),
            critical_alerts=critical_alerts
        )

        return results, summary

    def _calculate_energy_loss(
        self,
        condition: TrapCondition,
        input_data: TrapDiagnosticInput
    ) -> float:
        """Calculate energy loss based on condition and trap parameters."""
        import math

        # Energy loss factors (kW per mm^2 orifice at 10 bar)
        loss_factors = {
            TrapCondition.OPERATING_NORMAL: 0.0,
            TrapCondition.FAILED_OPEN: 2.5,
            TrapCondition.FAILED_CLOSED: 0.0,
            TrapCondition.LEAKING: 0.75,
            TrapCondition.INTERMITTENT: 0.3,
            TrapCondition.COLD: 0.0,
            TrapCondition.UNKNOWN: 0.3,
        }

        base_factor = loss_factors.get(condition, 0.0)

        # Calculate orifice area
        radius = input_data.orifice_diameter_mm / 2
        area_mm2 = math.pi * radius * radius

        # Pressure correction
        pressure_factor = input_data.pressure_bar_g / 10.0

        energy_loss = base_factor * area_mm2 * pressure_factor

        return round(energy_loss, 2)

    def _determine_alert_level(self, classification: ClassificationResult) -> AlertLevel:
        """Determine alert level from classification."""
        condition = classification.condition
        confidence = classification.confidence_score

        if condition == TrapCondition.FAILED_OPEN:
            return AlertLevel.CRITICAL
        elif condition == TrapCondition.FAILED_CLOSED:
            return AlertLevel.WARNING if confidence > 0.7 else AlertLevel.INFO
        elif condition == TrapCondition.LEAKING:
            if confidence > 0.8:
                return AlertLevel.WARNING
            return AlertLevel.INFO
        elif condition == TrapCondition.OPERATING_NORMAL:
            return AlertLevel.NONE
        else:
            return AlertLevel.INFO

    def _calculate_provenance_hash(
        self,
        input_data: TrapDiagnosticInput,
        classification: ClassificationResult,
        energy_loss: float
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        data = {
            "trap_id": input_data.trap_id,
            "acoustic_db": input_data.acoustic_amplitude_db,
            "inlet_temp": input_data.inlet_temp_c,
            "outlet_temp": input_data.outlet_temp_c,
            "condition": classification.condition.value,
            "confidence": classification.confidence_score,
            "energy_loss_kw": energy_loss
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def get_status(self) -> Dict[str, Any]:
        """Get agent status and statistics."""
        return {
            "agent_id": self.config.agent_id,
            "agent_name": self.config.agent_name,
            "version": self.config.version,
            "mode": self.config.mode.value,
            "status": "active",
            "statistics": {
                "diagnostic_count": self._diagnostic_count,
                "alert_count": self._alert_count
            },
            "components": {
                "classifier": self._classifier.get_statistics(),
                "explainer": self._explainer.get_statistics()
            }
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "TrapcatcherAgent",
    "AgentConfig",
    "AgentMode",
    "TrapDiagnosticInput",
    "DiagnosticOutput",
    "FleetSummary",
    "AlertLevel",
]
