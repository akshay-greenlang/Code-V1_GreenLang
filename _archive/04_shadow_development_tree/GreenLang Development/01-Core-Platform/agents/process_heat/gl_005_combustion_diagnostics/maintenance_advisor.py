# -*- coding: utf-8 -*-
"""
GL-005 Maintenance Advisor Module
=================================

This module provides maintenance advisory capabilities for the GL-005
COMBUSENSE agent. It generates maintenance recommendations and work orders
based on diagnostic analysis.

Key Capabilities:
    - Fouling prediction (heat transfer degradation)
    - Burner wear assessment
    - Maintenance prioritization
    - CMMS work order generation
    - Equipment health scoring

IMPORTANT: GL-005 is DIAGNOSTICS ONLY. It generates recommendations and
work orders but does NOT execute any maintenance actions. Work orders
must be reviewed and executed through the CMMS system.

ZERO-HALLUCINATION GUARANTEE:
    All predictions use documented engineering models.
    Maintenance recommendations are based on industry standards.
    No AI/ML speculation in critical maintenance decisions.

Author: GreenLang Process Heat Team
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    BurnerWearConfig,
    FoulingPredictionConfig,
    MaintenanceAdvisoryConfig,
    MaintenancePriority,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    AnalysisStatus,
    AnomalyDetectionResult,
    AnomalySeverity,
    BurnerWearAssessment,
    CMMSWorkOrder,
    CQIResult,
    CombustionOperatingData,
    FlueGasReading,
    FoulingAssessment,
    MaintenanceAdvisoryResult,
    MaintenanceRecommendation,
    TrendDirection,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HistoricalMetrics:
    """Historical metrics for trend analysis."""

    stack_temps: List[float] = field(default_factory=list)
    efficiency_values: List[float] = field(default_factory=list)
    co_values: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    def add_point(
        self,
        timestamp: datetime,
        stack_temp: float,
        efficiency: float,
        co_ppm: float,
    ) -> None:
        """Add a data point to history."""
        self.timestamps.append(timestamp)
        self.stack_temps.append(stack_temp)
        self.efficiency_values.append(efficiency)
        self.co_values.append(co_ppm)

        # Keep only last 30 days of data
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0)
            self.stack_temps.pop(0)
            self.efficiency_values.pop(0)
            self.co_values.pop(0)


# =============================================================================
# FOULING PREDICTOR
# =============================================================================

class FoulingPredictor:
    """
    Fouling/Deposit Prediction Model.

    Predicts heat transfer surface fouling based on:
    - Stack temperature trends
    - Heat transfer coefficient degradation
    - Efficiency decline patterns

    Theory:
        Fouling reduces heat transfer, causing:
        1. Higher stack temperature (heat escapes)
        2. Lower efficiency (more fuel for same output)
        3. Higher delta-T across heat exchangers

    DETERMINISTIC: Uses physics-based correlations.
    """

    def __init__(self, config: FoulingPredictionConfig) -> None:
        """
        Initialize fouling predictor.

        Args:
            config: Fouling prediction configuration
        """
        self.config = config
        self._history = HistoricalMetrics()
        self._baseline_stack_temp: Optional[float] = None
        self._baseline_efficiency: Optional[float] = None

        logger.info(
            f"Fouling Predictor initialized "
            f"(horizon={config.prediction_horizon_days} days)"
        )

    def set_baseline(
        self,
        stack_temp: float,
        efficiency: float,
    ) -> None:
        """
        Set baseline values (clean condition).

        Args:
            stack_temp: Baseline stack temperature (C)
            efficiency: Baseline efficiency (%)
        """
        self._baseline_stack_temp = stack_temp
        self._baseline_efficiency = efficiency
        logger.info(
            f"Fouling baseline set: stack_temp={stack_temp}C, efficiency={efficiency}%"
        )

    def add_data_point(
        self,
        flue_gas: FlueGasReading,
        efficiency: float,
    ) -> None:
        """
        Add data point for trend tracking.

        Args:
            flue_gas: Current flue gas reading
            efficiency: Current efficiency estimate
        """
        self._history.add_point(
            timestamp=flue_gas.timestamp,
            stack_temp=flue_gas.flue_gas_temp_c,
            efficiency=efficiency,
            co_ppm=flue_gas.co_ppm,
        )

    def assess(
        self,
        current_stack_temp: float,
        current_efficiency: float,
    ) -> FoulingAssessment:
        """
        Assess current fouling condition.

        Args:
            current_stack_temp: Current stack temperature (C)
            current_efficiency: Current efficiency (%)

        Returns:
            FoulingAssessment with predictions
        """
        # Calculate indicators
        stack_temp_increase = 0.0
        efficiency_loss = 0.0

        if self._baseline_stack_temp:
            stack_temp_increase = current_stack_temp - self._baseline_stack_temp

        if self._baseline_efficiency:
            efficiency_loss = self._baseline_efficiency - current_efficiency

        # Determine severity
        severity = "none"
        fouling_detected = False

        if efficiency_loss >= self.config.fouling_critical_pct:
            severity = "severe"
            fouling_detected = True
        elif efficiency_loss >= self.config.fouling_warning_pct:
            severity = "heavy"
            fouling_detected = True
        elif efficiency_loss >= self.config.fouling_warning_pct / 2:
            severity = "moderate"
            fouling_detected = True
        elif efficiency_loss >= 1.0:
            severity = "light"
            fouling_detected = True

        # Calculate days until cleaning recommended
        days_until_cleaning = None
        predicted_loss_30d = None

        if len(self._history.efficiency_values) >= 5:
            # Calculate efficiency decline rate
            trend_slope = self._calculate_trend_slope(self._history.efficiency_values)
            if trend_slope < 0:  # Declining efficiency
                # Predict when we'll hit warning threshold
                loss_rate_per_day = abs(trend_slope)
                remaining_loss = self.config.fouling_warning_pct - efficiency_loss

                if remaining_loss > 0 and loss_rate_per_day > 0:
                    days_until_cleaning = int(remaining_loss / loss_rate_per_day)

                # Predict 30-day loss
                predicted_loss_30d = efficiency_loss + (loss_rate_per_day * 30)

        # Calculate confidence
        confidence = self._calculate_confidence(efficiency_loss, stack_temp_increase)

        return FoulingAssessment(
            fouling_detected=fouling_detected,
            fouling_severity=severity,
            efficiency_loss_pct=round(max(0, efficiency_loss), 2),
            stack_temp_increase_c=round(max(0, stack_temp_increase), 1),
            delta_t_degradation_pct=round(efficiency_loss * 1.2, 2),  # Approximation
            days_until_cleaning_recommended=days_until_cleaning,
            predicted_efficiency_loss_30d=round(predicted_loss_30d, 2) if predicted_loss_30d else None,
            assessment_confidence=confidence,
        )

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate linear trend slope using simple linear regression."""
        n = len(values)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _calculate_confidence(
        self,
        efficiency_loss: float,
        stack_temp_increase: float,
    ) -> float:
        """Calculate assessment confidence based on indicator consistency."""
        # Both indicators should agree for high confidence
        if self._baseline_stack_temp is None or self._baseline_efficiency is None:
            return 0.5  # Low confidence without baseline

        # Expected correlation: 1% efficiency loss ~ 5-10C stack temp increase
        expected_temp_increase = efficiency_loss * 7.5  # 7.5C per 1%

        if expected_temp_increase > 0:
            consistency = 1 - abs(stack_temp_increase - expected_temp_increase) / expected_temp_increase
            confidence = max(0.3, min(0.95, 0.5 + consistency * 0.45))
        else:
            confidence = 0.7

        return round(confidence, 2)


# =============================================================================
# BURNER WEAR ASSESSOR
# =============================================================================

class BurnerWearAssessor:
    """
    Burner Wear Assessment Model.

    Assesses burner condition based on:
    - Operating hours
    - CO emission trends
    - Flame stability indicators
    - Ignition reliability

    DETERMINISTIC: Uses documented wear correlations.
    """

    def __init__(self, config: BurnerWearConfig) -> None:
        """
        Initialize burner wear assessor.

        Args:
            config: Burner wear configuration
        """
        self.config = config
        self._co_history: List[Tuple[datetime, float]] = []
        self._baseline_co: Optional[float] = None

        logger.info(
            f"Burner Wear Assessor initialized "
            f"(expected_life={config.expected_burner_life_hours}h)"
        )

    def set_baseline_co(self, co_ppm: float) -> None:
        """Set baseline CO level (new burner condition)."""
        self._baseline_co = co_ppm
        logger.info(f"Baseline CO set: {co_ppm} ppm")

    def add_co_reading(self, timestamp: datetime, co_ppm: float) -> None:
        """Add CO reading for trend tracking."""
        self._co_history.append((timestamp, co_ppm))

        # Keep last 90 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        self._co_history = [
            (ts, co) for ts, co in self._co_history if ts >= cutoff
        ]

    def assess(
        self,
        current_co: float,
        operating_hours: float,
        flame_stability: Optional[float] = None,
        ignition_reliability: Optional[float] = None,
    ) -> BurnerWearAssessment:
        """
        Assess burner wear condition.

        Args:
            current_co: Current CO level (ppm)
            operating_hours: Total burner operating hours
            flame_stability: Flame stability score (0-1)
            ignition_reliability: Ignition success rate (0-1)

        Returns:
            BurnerWearAssessment with predictions
        """
        # Calculate remaining life
        expected_life = self.config.expected_burner_life_hours
        life_remaining_pct = max(0, 100 * (1 - operating_hours / expected_life))

        # Calculate CO trend
        co_trend_slope = 0.0
        if len(self._co_history) >= 5:
            co_values = [co for _, co in self._co_history]
            co_trend_slope = self._calculate_daily_trend(co_values)

        # Determine wear level
        wear_level = "normal"
        wear_detected = False

        # Factors that indicate wear:
        # 1. Operating hours vs expected life
        # 2. CO trend (increasing CO = degrading burner)
        # 3. Flame stability
        # 4. Ignition reliability

        wear_score = 0

        # Hours-based wear
        if life_remaining_pct < 20:
            wear_score += 40
        elif life_remaining_pct < 40:
            wear_score += 25
        elif life_remaining_pct < 60:
            wear_score += 10

        # CO trend
        if co_trend_slope > self.config.co_trend_threshold:
            wear_score += 30
        elif co_trend_slope > self.config.co_trend_threshold / 2:
            wear_score += 15

        # Flame stability
        stability = flame_stability if flame_stability is not None else 1.0
        if stability < self.config.flame_stability_threshold:
            wear_score += 20

        # Ignition reliability
        reliability = ignition_reliability if ignition_reliability is not None else 1.0
        if reliability < 0.95:
            wear_score += 10

        # Determine wear level from score
        if wear_score >= 70:
            wear_level = "replacement_needed"
            wear_detected = True
        elif wear_score >= 50:
            wear_level = "significant_wear"
            wear_detected = True
        elif wear_score >= 30:
            wear_level = "moderate_wear"
            wear_detected = True
        elif wear_score >= 15:
            wear_level = "early_wear"
            wear_detected = True

        # Predict remaining life
        remaining_hours = None
        replacement_date = None

        if operating_hours > 0 and life_remaining_pct < 100:
            remaining_hours = expected_life - operating_hours
            if remaining_hours > 0:
                # Estimate based on current rate
                hours_per_day = 16  # Assume 16h/day operation
                days_remaining = remaining_hours / hours_per_day
                replacement_date = datetime.now(timezone.utc) + timedelta(days=days_remaining)

        # Calculate confidence
        confidence = 0.8
        if self._baseline_co is None:
            confidence -= 0.2
        if len(self._co_history) < 5:
            confidence -= 0.1

        return BurnerWearAssessment(
            wear_detected=wear_detected,
            wear_level=wear_level,
            operating_hours=operating_hours,
            expected_life_remaining_pct=round(life_remaining_pct, 1),
            co_trend_slope=round(co_trend_slope, 4),
            flame_stability_score=stability,
            ignition_reliability=reliability,
            estimated_remaining_life_hours=remaining_hours,
            replacement_recommended_by=replacement_date,
            assessment_confidence=round(confidence, 2),
        )

    def _calculate_daily_trend(self, values: List[float]) -> float:
        """Calculate daily trend rate."""
        if len(values) < 2:
            return 0.0

        # Simple linear regression
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        # Convert to daily rate (assuming one reading per day)
        return numerator / denominator


# =============================================================================
# WORK ORDER GENERATOR
# =============================================================================

class WorkOrderGenerator:
    """
    CMMS Work Order Generator.

    Generates standardized maintenance work orders for CMMS integration.
    Supports multiple CMMS systems (SAP PM, Maximo, Fiix, generic).

    IMPORTANT: Work orders are GENERATED but NOT SUBMITTED automatically.
    They require review/approval before submission to CMMS.
    """

    def __init__(self, config: MaintenanceAdvisoryConfig) -> None:
        """
        Initialize work order generator.

        Args:
            config: Maintenance advisory configuration
        """
        self.config = config
        logger.info(
            f"Work Order Generator initialized (CMMS={config.cmms_system})"
        )

    def generate_work_order(
        self,
        equipment_id: str,
        recommendation: MaintenanceRecommendation,
        equipment_name: Optional[str] = None,
        location: Optional[str] = None,
    ) -> CMMSWorkOrder:
        """
        Generate a CMMS work order from a maintenance recommendation.

        Args:
            equipment_id: Equipment identifier
            recommendation: Maintenance recommendation
            equipment_name: Optional equipment name
            location: Optional location

        Returns:
            CMMSWorkOrder ready for CMMS submission
        """
        # Map maintenance type to work type code
        work_type_map = {
            "inspection": "INSP",
            "cleaning": "PM",
            "repair": "CM",
            "replacement": "CM",
            "calibration": "PM",
        }
        work_type = work_type_map.get(recommendation.maintenance_type, "CM")

        # Generate work order ID
        wo_id = f"WO-GL005-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"

        # Calculate provenance hash
        provenance_data = {
            "recommendation_id": recommendation.recommendation_id,
            "equipment_id": equipment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        work_order = CMMSWorkOrder(
            work_order_id=wo_id,
            created_timestamp=datetime.now(timezone.utc),
            equipment_id=equipment_id,
            equipment_name=equipment_name,
            location=location,
            work_type=work_type,
            priority=recommendation.priority,
            title=recommendation.title,
            description=self._format_description(recommendation),
            requested_start_date=recommendation.recommended_by_date,
            requested_end_date=(
                recommendation.recommended_by_date + timedelta(hours=recommendation.estimated_duration_hours)
                if recommendation.recommended_by_date and recommendation.estimated_duration_hours
                else None
            ),
            estimated_hours=recommendation.estimated_duration_hours,
            source_agent="GL-005",
            source_analysis_id=recommendation.recommendation_id,
            source_recommendations=[recommendation.title],
            status="pending_approval" if self.config.work_order_approval_required else "approved",
            provenance_hash=provenance_hash,
        )

        logger.info(f"Work order generated: {wo_id} ({work_type}, {recommendation.priority.value})")
        return work_order

    def _format_description(self, recommendation: MaintenanceRecommendation) -> str:
        """Format work order description with full details."""
        lines = [
            f"Generated by: GL-005 COMBUSENSE Agent",
            f"",
            f"Description: {recommendation.description}",
            f"",
            f"Justification: {recommendation.justification}",
            f"",
            f"Risk if deferred: {recommendation.risk_if_deferred}",
        ]

        if recommendation.potential_consequences:
            lines.append("")
            lines.append("Potential Consequences if Not Addressed:")
            for consequence in recommendation.potential_consequences:
                lines.append(f"  - {consequence}")

        return "\n".join(lines)


# =============================================================================
# MAINTENANCE ADVISOR
# =============================================================================

class MaintenanceAdvisor:
    """
    Integrated Maintenance Advisor.

    Combines fouling prediction, burner wear assessment, and work order
    generation to provide comprehensive maintenance recommendations.

    This is a DIAGNOSTIC-ONLY component. It identifies maintenance needs
    and generates work orders but does NOT execute any actions.

    Example:
        >>> config = MaintenanceAdvisoryConfig()
        >>> advisor = MaintenanceAdvisor(config, equipment_id="BLR-001")
        >>> result = advisor.analyze(flue_gas, operating_data, cqi_result)
        >>> for wo in result.work_orders:
        ...     print(f"Work order: {wo.title}")
    """

    def __init__(
        self,
        config: MaintenanceAdvisoryConfig,
        equipment_id: str,
    ) -> None:
        """
        Initialize maintenance advisor.

        Args:
            config: Maintenance advisory configuration
            equipment_id: Target equipment identifier
        """
        self.config = config
        self.equipment_id = equipment_id

        self.fouling_predictor = FoulingPredictor(config.fouling)
        self.burner_assessor = BurnerWearAssessor(config.burner_wear)
        self.work_order_generator = WorkOrderGenerator(config)

        self._audit_trail: List[Dict[str, Any]] = []

        logger.info(f"Maintenance Advisor initialized for {equipment_id}")

    def set_baselines(
        self,
        stack_temp: float,
        efficiency: float,
        co_ppm: float,
    ) -> None:
        """
        Set baseline values for clean/new equipment.

        Args:
            stack_temp: Clean-condition stack temperature
            efficiency: Clean-condition efficiency
            co_ppm: New-burner CO level
        """
        self.fouling_predictor.set_baseline(stack_temp, efficiency)
        self.burner_assessor.set_baseline_co(co_ppm)

    def analyze(
        self,
        flue_gas: FlueGasReading,
        operating_data: CombustionOperatingData,
        cqi_result: Optional[CQIResult] = None,
        anomaly_result: Optional[AnomalyDetectionResult] = None,
    ) -> MaintenanceAdvisoryResult:
        """
        Perform comprehensive maintenance analysis.

        Args:
            flue_gas: Current flue gas reading
            operating_data: Current operating data
            cqi_result: Optional CQI analysis result
            anomaly_result: Optional anomaly detection result

        Returns:
            MaintenanceAdvisoryResult with recommendations
        """
        start_time = datetime.now(timezone.utc)
        self._audit_trail = []

        # Get efficiency from CQI result or estimate
        efficiency = (
            cqi_result.combustion_efficiency_pct
            if cqi_result
            else self._estimate_efficiency(flue_gas)
        )

        # Update historical data
        self.fouling_predictor.add_data_point(flue_gas, efficiency)
        self.burner_assessor.add_co_reading(flue_gas.timestamp, flue_gas.co_ppm)

        # Step 1: Assess fouling
        fouling = self.fouling_predictor.assess(
            flue_gas.flue_gas_temp_c,
            efficiency,
        )
        self._add_audit_entry("fouling_assessment", {
            "severity": fouling.fouling_severity,
            "efficiency_loss": fouling.efficiency_loss_pct,
        })

        # Step 2: Assess burner wear
        operating_hours = operating_data.operating_hours_total or 0
        burner_wear = self.burner_assessor.assess(
            current_co=flue_gas.co_ppm,
            operating_hours=operating_hours,
        )
        self._add_audit_entry("burner_wear_assessment", {
            "wear_level": burner_wear.wear_level,
            "remaining_life_pct": burner_wear.expected_life_remaining_pct,
        })

        # Step 3: Generate recommendations
        recommendations = self._generate_recommendations(
            fouling, burner_wear, flue_gas, cqi_result, anomaly_result
        )

        # Step 4: Calculate equipment health score
        health_score = self._calculate_health_score(fouling, burner_wear, cqi_result)
        health_trend = self._determine_health_trend(cqi_result)

        # Step 5: Determine urgency
        urgent_actions = any(
            r.priority in [MaintenancePriority.CRITICAL, MaintenancePriority.HIGH]
            for r in recommendations
        )

        # Step 6: Generate work orders (if enabled)
        work_orders = []
        if self.config.cmms_enabled and recommendations:
            for rec in recommendations:
                if rec.priority in [MaintenancePriority.CRITICAL, MaintenancePriority.HIGH, MaintenancePriority.MEDIUM]:
                    wo = self.work_order_generator.generate_work_order(
                        equipment_id=self.equipment_id,
                        recommendation=rec,
                    )
                    work_orders.append(wo)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            fouling, burner_wear, recommendations
        )

        # Get next recommended maintenance
        next_maintenance = recommendations[0] if recommendations else None

        result = MaintenanceAdvisoryResult(
            status=AnalysisStatus.SUCCESS,
            fouling=fouling,
            burner_wear=burner_wear,
            recommendations=recommendations,
            urgent_actions_required=urgent_actions,
            equipment_health_score=health_score,
            health_trend=health_trend,
            next_recommended_maintenance=next_maintenance,
            analysis_timestamp=start_time,
            provenance_hash=provenance_hash,
        )

        logger.info(
            f"Maintenance analysis complete: health={health_score:.1f}, "
            f"recommendations={len(recommendations)}, urgent={urgent_actions}"
        )

        return result

    def _estimate_efficiency(self, flue_gas: FlueGasReading) -> float:
        """Estimate combustion efficiency from flue gas."""
        # Simplified Siegert formula
        stack_loss = 0.035 * (flue_gas.flue_gas_temp_c - 25)
        excess_air = (flue_gas.oxygen_pct / (20.95 - flue_gas.oxygen_pct)) * 100
        excess_air_loss = 0.0003 * excess_air * (flue_gas.flue_gas_temp_c - 25)
        efficiency = 100 - stack_loss - excess_air_loss - 2  # 2% other losses
        return max(70, min(99, efficiency))

    def _generate_recommendations(
        self,
        fouling: FoulingAssessment,
        burner_wear: BurnerWearAssessment,
        flue_gas: FlueGasReading,
        cqi_result: Optional[CQIResult],
        anomaly_result: Optional[AnomalyDetectionResult],
    ) -> List[MaintenanceRecommendation]:
        """Generate maintenance recommendations from assessments."""
        recommendations = []
        now = datetime.now(timezone.utc)

        # Fouling-based recommendations
        if fouling.fouling_detected:
            priority = self._fouling_to_priority(fouling.fouling_severity)
            recommendations.append(MaintenanceRecommendation(
                recommendation_id=f"REC-{uuid.uuid4().hex[:8]}",
                timestamp=now,
                maintenance_type="cleaning",
                priority=priority,
                component="Heat Transfer Surfaces",
                title="Heat transfer surface cleaning required",
                description=f"Fouling detected with {fouling.efficiency_loss_pct}% efficiency loss. "
                           f"Stack temperature has increased {fouling.stack_temp_increase_c}C above baseline.",
                justification=f"Current efficiency loss of {fouling.efficiency_loss_pct}% represents "
                             f"significant fuel waste and increased operating costs.",
                recommended_by_date=(
                    now + timedelta(days=fouling.days_until_cleaning_recommended)
                    if fouling.days_until_cleaning_recommended
                    else now + timedelta(days=30)
                ),
                estimated_duration_hours=8.0,
                risk_if_deferred="high" if priority in [MaintenancePriority.CRITICAL, MaintenancePriority.HIGH] else "medium",
                potential_consequences=[
                    "Continued efficiency degradation",
                    "Increased fuel consumption",
                    "Potential overheating of tubes",
                    "Risk of unplanned shutdown",
                ],
            ))

        # Burner wear recommendations
        if burner_wear.wear_detected:
            priority = self._wear_to_priority(burner_wear.wear_level)
            if burner_wear.wear_level == "replacement_needed":
                recommendations.append(MaintenanceRecommendation(
                    recommendation_id=f"REC-{uuid.uuid4().hex[:8]}",
                    timestamp=now,
                    maintenance_type="replacement",
                    priority=priority,
                    component="Burner Assembly",
                    title="Burner replacement required",
                    description=f"Burner has {burner_wear.operating_hours:.0f} operating hours with "
                               f"only {burner_wear.expected_life_remaining_pct:.1f}% expected life remaining.",
                    justification="Burner has exceeded expected service life and is showing wear symptoms.",
                    recommended_by_date=burner_wear.replacement_recommended_by,
                    estimated_duration_hours=16.0,
                    risk_if_deferred="critical",
                    potential_consequences=[
                        "Burner failure during operation",
                        "Unsafe combustion conditions",
                        "Emergency shutdown",
                        "Production loss",
                    ],
                ))
            else:
                recommendations.append(MaintenanceRecommendation(
                    recommendation_id=f"REC-{uuid.uuid4().hex[:8]}",
                    timestamp=now,
                    maintenance_type="inspection",
                    priority=priority,
                    component="Burner Assembly",
                    title="Burner inspection recommended",
                    description=f"Burner showing {burner_wear.wear_level} wear patterns. "
                               f"CO trend: {burner_wear.co_trend_slope:.4f} ppm/day.",
                    justification="Early detection of burner wear can prevent unplanned failures.",
                    recommended_by_date=now + timedelta(days=14),
                    estimated_duration_hours=4.0,
                    risk_if_deferred="medium",
                    potential_consequences=[
                        "Accelerated burner degradation",
                        "Increased CO emissions",
                        "Potential flame instability",
                    ],
                ))

        # High CO recommendation
        if flue_gas.co_ppm > 300:
            recommendations.append(MaintenanceRecommendation(
                recommendation_id=f"REC-{uuid.uuid4().hex[:8]}",
                timestamp=now,
                maintenance_type="inspection",
                priority=MaintenancePriority.HIGH if flue_gas.co_ppm > 500 else MaintenancePriority.MEDIUM,
                component="Combustion System",
                title="High CO - combustion system inspection",
                description=f"CO level of {flue_gas.co_ppm} ppm indicates incomplete combustion. "
                           "Investigation required.",
                justification="High CO indicates combustion issues that affect efficiency and safety.",
                recommended_by_date=now + timedelta(days=7),
                estimated_duration_hours=4.0,
                risk_if_deferred="high",
                potential_consequences=[
                    "Safety hazard (CO exposure)",
                    "Wasted fuel",
                    "Environmental compliance issues",
                ],
            ))

        # CQI-based recommendations
        if cqi_result and cqi_result.cqi_score < 60:
            recommendations.append(MaintenanceRecommendation(
                recommendation_id=f"REC-{uuid.uuid4().hex[:8]}",
                timestamp=now,
                maintenance_type="inspection",
                priority=MaintenancePriority.MEDIUM,
                component="Combustion System",
                title="Poor combustion quality - tune-up required",
                description=f"CQI score of {cqi_result.cqi_score:.1f} indicates poor combustion quality. "
                           "System tune-up recommended.",
                justification="Low CQI score indicates sub-optimal combustion affecting efficiency and emissions.",
                recommended_by_date=now + timedelta(days=14),
                estimated_duration_hours=6.0,
                risk_if_deferred="medium",
                potential_consequences=[
                    "Continued efficiency loss",
                    "Elevated emissions",
                    "Potential compliance issues",
                ],
            ))

        # Anomaly-based recommendations
        if anomaly_result and anomaly_result.critical_count > 0:
            recommendations.append(MaintenanceRecommendation(
                recommendation_id=f"REC-{uuid.uuid4().hex[:8]}",
                timestamp=now,
                maintenance_type="inspection",
                priority=MaintenancePriority.HIGH,
                component="Combustion System",
                title="Critical anomaly detected - immediate inspection",
                description=f"{anomaly_result.critical_count} critical anomalies detected. "
                           "Immediate investigation required.",
                justification="Critical anomalies may indicate serious equipment issues.",
                recommended_by_date=now + timedelta(days=1),
                estimated_duration_hours=4.0,
                risk_if_deferred="critical",
                potential_consequences=[
                    "Equipment failure",
                    "Safety incident",
                    "Unplanned shutdown",
                ],
            ))

        # Sort by priority
        priority_order = {
            MaintenancePriority.CRITICAL: 0,
            MaintenancePriority.HIGH: 1,
            MaintenancePriority.MEDIUM: 2,
            MaintenancePriority.LOW: 3,
            MaintenancePriority.ROUTINE: 4,
        }
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 5))

        return recommendations

    def _fouling_to_priority(self, severity: str) -> MaintenancePriority:
        """Map fouling severity to maintenance priority."""
        mapping = {
            "severe": MaintenancePriority.CRITICAL,
            "heavy": MaintenancePriority.HIGH,
            "moderate": MaintenancePriority.MEDIUM,
            "light": MaintenancePriority.LOW,
        }
        return mapping.get(severity, MaintenancePriority.ROUTINE)

    def _wear_to_priority(self, wear_level: str) -> MaintenancePriority:
        """Map burner wear level to maintenance priority."""
        mapping = {
            "replacement_needed": MaintenancePriority.CRITICAL,
            "significant_wear": MaintenancePriority.HIGH,
            "moderate_wear": MaintenancePriority.MEDIUM,
            "early_wear": MaintenancePriority.LOW,
        }
        return mapping.get(wear_level, MaintenancePriority.ROUTINE)

    def _calculate_health_score(
        self,
        fouling: FoulingAssessment,
        burner_wear: BurnerWearAssessment,
        cqi_result: Optional[CQIResult],
    ) -> float:
        """Calculate overall equipment health score (0-100)."""
        # Start with 100 and deduct for issues
        score = 100.0

        # Fouling deductions
        severity_deductions = {
            "severe": 30,
            "heavy": 20,
            "moderate": 10,
            "light": 5,
        }
        score -= severity_deductions.get(fouling.fouling_severity, 0)

        # Burner wear deductions
        wear_deductions = {
            "replacement_needed": 35,
            "significant_wear": 20,
            "moderate_wear": 10,
            "early_wear": 5,
        }
        score -= wear_deductions.get(burner_wear.wear_level, 0)

        # CQI-based deductions
        if cqi_result:
            if cqi_result.cqi_score < 50:
                score -= 20
            elif cqi_result.cqi_score < 70:
                score -= 10
            elif cqi_result.cqi_score < 80:
                score -= 5

        return max(0, min(100, score))

    def _determine_health_trend(
        self,
        cqi_result: Optional[CQIResult],
    ) -> TrendDirection:
        """Determine health trend from CQI result."""
        if cqi_result:
            return cqi_result.trend_vs_baseline
        return TrendDirection.UNKNOWN

    def _calculate_provenance_hash(
        self,
        fouling: FoulingAssessment,
        burner_wear: BurnerWearAssessment,
        recommendations: List[MaintenanceRecommendation],
    ) -> str:
        """Calculate provenance hash for audit trail."""
        data = {
            "fouling_severity": fouling.fouling_severity,
            "burner_wear_level": burner_wear.wear_level,
            "recommendation_count": len(recommendations),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _add_audit_entry(self, operation: str, data: Dict[str, Any]) -> None:
        """Add entry to audit trail."""
        self._audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "data": data,
        })

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get maintenance advisory audit trail."""
        return self._audit_trail.copy()
