# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Main Orchestrator

Central orchestrator for the Insulation Scanning & Thermal Assessment agent.
Coordinates thermal analysis, hot spot detection, heat loss calculations,
condition assessment, and repair recommendation workflows.

The orchestrator ensures deterministic, reproducible operation with
full provenance tracking and audit logging for all calculations.

Example:
    >>> from core import InsulscanOrchestrator, InsulscanSettings
    >>> settings = InsulscanSettings()
    >>> orchestrator = InsulscanOrchestrator(settings)
    >>> result = await orchestrator.analyze_insulation(asset, measurements)

Author: GreenLang GL-015 INSULSCAN
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    InsulscanSettings,
    get_settings,
    InsulationType,
    SurfaceType,
    HotSpotSeverity,
    ConditionSeverity,
    RepairPriority,
    RepairType,
    DataQuality,
    DEFAULT_THERMAL_CONDUCTIVITY,
    DEFAULT_EMISSIVITY,
)
from .schemas import (
    InsulationAsset,
    ThermalMeasurement,
    HotSpotDetection,
    InsulationCondition,
    HeatLossResult,
    RepairRecommendation,
    AnalysisResult,
    AnalysisStatus,
    DamageType,
)
from .seed_manager import SeedManager, SeedDomain, get_seed_manager

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES FOR INTERNAL STATE
# =============================================================================


@dataclass
class CalculationEvent:
    """Event record for a single calculation."""

    event_id: str
    timestamp: datetime
    calculation_type: str
    asset_id: str
    inputs_hash: str
    outputs_hash: str
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "calculation_type": self.calculation_type,
            "asset_id": self.asset_id,
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class OrchestratorStatus:
    """Current status of the orchestrator."""

    agent_id: str = "GL-015"
    agent_name: str = "INSULSCAN"
    version: str = "1.0.0"
    status: str = "running"
    health: str = "healthy"
    uptime_seconds: float = 0.0
    analyses_performed: int = 0
    analyses_successful: int = 0
    hot_spots_detected: int = 0
    last_analysis_time: Optional[datetime] = None
    error_count: int = 0


# =============================================================================
# MAIN ORCHESTRATOR CLASS
# =============================================================================


class InsulscanOrchestrator:
    """
    Main orchestrator for GL-015 INSULSCAN agent.

    Coordinates all insulation analysis workflows including:
    - Thermal measurement processing
    - Hot spot detection and classification
    - Heat loss calculations (deterministic physics formulas)
    - Insulation condition assessment
    - Repair recommendation generation

    Ensures deterministic, reproducible operation with full
    provenance tracking and audit logging.

    Example:
        >>> orchestrator = InsulscanOrchestrator()
        >>> result = await orchestrator.analyze_insulation(asset, measurements)
        >>> print(f"Condition score: {result.condition.condition_score}")
        >>> print(f"Heat loss: {result.heat_loss.heat_loss_w} W")

    Attributes:
        settings: Agent configuration settings
        seed_manager: Deterministic seed manager for ML reproducibility
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        settings: Optional[InsulscanSettings] = None,
    ) -> None:
        """
        Initialize the INSULSCAN orchestrator.

        Args:
            settings: Agent configuration settings. If None, uses defaults.
        """
        self.settings = settings or get_settings()
        self._start_time = datetime.now(timezone.utc)

        # Initialize seed manager for reproducible ML predictions
        self.seed_manager = get_seed_manager()
        self.seed_manager.set_global_seed(
            self.settings.ml.random_seed,
            caller="InsulscanOrchestrator.__init__"
        )

        # Statistics tracking
        self._analyses_count = 0
        self._successful_count = 0
        self._hot_spots_total = 0
        self._error_count = 0
        self._last_analysis_time: Optional[datetime] = None

        # Calculation event log for audit trail
        self._calculation_events: List[CalculationEvent] = []
        self._max_event_history = 10000

        # Reference data cache
        self._baseline_cache: Dict[str, HeatLossResult] = {}

        logger.info(
            f"GL-015 INSULSCAN orchestrator initialized: "
            f"version={self.VERSION}, "
            f"deterministic_mode={self.settings.deterministic_mode}, "
            f"seed={self.settings.ml.random_seed}"
        )

    # =========================================================================
    # PUBLIC API - ANALYSIS METHODS
    # =========================================================================

    async def analyze_insulation(
        self,
        asset: InsulationAsset,
        measurements: List[ThermalMeasurement],
        include_recommendations: bool = True,
    ) -> AnalysisResult:
        """
        Perform complete analysis of an insulation asset.

        Executes thermal analysis, hot spot detection, heat loss calculation,
        condition assessment, and optionally generates repair recommendations.

        ZERO-HALLUCINATION: All calculations use deterministic formulas.
        No LLM calls for numeric values.

        Args:
            asset: Insulation asset definition
            measurements: List of thermal measurements
            include_recommendations: Generate repair recommendations

        Returns:
            AnalysisResult with condition, heat loss, hot spots, and recommendations

        Example:
            >>> result = await orchestrator.analyze_insulation(asset, measurements)
            >>> print(f"Score: {result.condition.condition_score}")
        """
        start_time = datetime.now(timezone.utc)
        self._analyses_count += 1
        asset_id = asset.asset_id

        logger.info(
            f"Starting analysis: asset_id={asset_id}, "
            f"measurements={len(measurements)}"
        )

        warnings: List[str] = []
        errors: List[str] = []

        try:
            # Validate input measurements
            valid_measurements = self._filter_valid_measurements(measurements)
            if len(valid_measurements) < len(measurements):
                warnings.append(
                    f"Filtered {len(measurements) - len(valid_measurements)} "
                    f"low-quality measurements"
                )

            if not valid_measurements:
                raise ValueError("No valid measurements provided")

            # Step 1: Detect hot spots
            hot_spots = await self.detect_hot_spots(
                asset,
                valid_measurements,
            )
            self._hot_spots_total += len(hot_spots)

            # Step 2: Calculate heat loss
            heat_loss = await self.calculate_heat_loss(
                asset,
                valid_measurements,
            )

            # Step 3: Assess insulation condition
            condition = await self._assess_condition(
                asset,
                valid_measurements,
                hot_spots,
                heat_loss,
            )

            # Step 4: Generate repair recommendation (if requested)
            recommendation = None
            if include_recommendations and self.settings.features.enable_repair_recommendations:
                recommendation = await self.generate_repair_recommendation(
                    asset,
                    condition,
                    heat_loss,
                )

            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Determine overall score and risk level
            overall_score = condition.condition_score
            if overall_score >= 75:
                risk_level = "low"
            elif overall_score >= 50:
                risk_level = "medium"
            elif overall_score >= 25:
                risk_level = "high"
            else:
                risk_level = "critical"

            action_required = (
                condition.severity in [ConditionSeverity.POOR, ConditionSeverity.CRITICAL, ConditionSeverity.FAILED]
                or condition.critical_hot_spots > 0
            )

            # Create result
            result = AnalysisResult(
                asset_id=asset_id,
                status=AnalysisStatus.COMPLETED,
                asset=asset,
                measurements=valid_measurements,
                condition=condition,
                heat_loss=heat_loss,
                hot_spots=hot_spots,
                recommendation=recommendation,
                overall_score=overall_score,
                risk_level=risk_level,
                action_required=action_required,
                processing_time_ms=processing_time,
                warnings=warnings,
                errors=errors,
                agent_version=self.VERSION,
            )

            # Compute provenance hashes
            result.input_hash = self._compute_hash({
                "asset_id": asset_id,
                "measurements_count": len(valid_measurements),
                "operating_temp": asset.operating_temp_c,
                "thickness_mm": asset.thickness_mm,
            })
            result.output_hash = self._compute_hash({
                "condition_score": condition.condition_score,
                "heat_loss_w": round(heat_loss.heat_loss_w, 2),
                "hot_spots": len(hot_spots),
            })

            # Update statistics
            self._successful_count += 1
            self._last_analysis_time = datetime.now(timezone.utc)

            # Log calculation event
            self._log_calculation_event(
                calculation_type="full_analysis",
                asset_id=asset_id,
                inputs_hash=result.input_hash,
                outputs_hash=result.output_hash,
                execution_time_ms=processing_time,
                success=True,
            )

            logger.info(
                f"Analysis complete: asset_id={asset_id}, "
                f"score={condition.condition_score}, "
                f"heat_loss={heat_loss.heat_loss_w:.1f}W, "
                f"hot_spots={len(hot_spots)}, "
                f"time={processing_time:.1f}ms"
            )

            return result

        except Exception as e:
            self._error_count += 1
            error_msg = str(e)
            logger.error(f"Analysis failed: asset_id={asset_id}, error={error_msg}")

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self._log_calculation_event(
                calculation_type="full_analysis",
                asset_id=asset_id,
                inputs_hash="",
                outputs_hash="",
                execution_time_ms=processing_time,
                success=False,
                error_message=error_msg,
            )

            return AnalysisResult(
                asset_id=asset_id,
                status=AnalysisStatus.FAILED,
                processing_time_ms=processing_time,
                errors=[error_msg],
                agent_version=self.VERSION,
            )

    async def detect_hot_spots(
        self,
        asset: InsulationAsset,
        measurements: List[ThermalMeasurement],
    ) -> List[HotSpotDetection]:
        """
        Detect thermal anomalies (hot spots) from measurements.

        Uses deterministic threshold-based detection and optionally
        ML-based pattern recognition.

        ZERO-HALLUCINATION: Classification uses fixed thresholds.
        ML is only used for optional pattern enhancement.

        Args:
            asset: Insulation asset
            measurements: Thermal measurements

        Returns:
            List of detected hot spots
        """
        if not self.settings.features.enable_hot_spot_detection:
            return []

        hot_spots: List[HotSpotDetection] = []

        # Calculate expected surface temperature
        expected_surface_temp = self._calculate_expected_surface_temp(asset)

        # Thresholds from settings
        warning_delta = self.settings.thermal.hot_spot_warning_delta_c
        critical_delta = self.settings.thermal.hot_spot_critical_delta_c
        emergency_delta = self.settings.thermal.hot_spot_emergency_delta_c
        safety_temp = self.settings.thermal.personnel_safety_temp_c

        for measurement in measurements:
            temp_diff = measurement.surface_temp_c - expected_surface_temp

            # Classify severity based on temperature differential
            if temp_diff >= emergency_delta:
                severity = HotSpotSeverity.EMERGENCY
            elif temp_diff >= critical_delta:
                severity = HotSpotSeverity.CRITICAL
            elif temp_diff >= warning_delta:
                severity = HotSpotSeverity.WARNING
            else:
                severity = HotSpotSeverity.NORMAL

            # Create hot spot detection if above warning threshold
            if severity != HotSpotSeverity.NORMAL:
                # Estimate affected area (simplified)
                area_m2 = self.settings.ml.min_hot_spot_area_m2

                # Determine probable cause based on characteristics
                probable_cause = self._infer_damage_cause(
                    temp_diff,
                    measurement,
                    asset,
                )

                # Estimate additional heat loss from hot spot
                # Q = h * A * delta_T (simplified)
                h_conv = 10.0 + 5.0 * measurement.wind_speed_ms
                additional_heat_loss = h_conv * area_m2 * temp_diff

                # Check personnel safety
                personnel_risk = measurement.surface_temp_c > safety_temp

                # Compute provenance hash
                hash_input = {
                    "asset_id": asset.asset_id,
                    "severity": severity.value,
                    "temp_diff": round(temp_diff, 2),
                    "peak_temp": round(measurement.surface_temp_c, 2),
                }
                provenance_hash = hashlib.sha256(
                    json.dumps(hash_input, sort_keys=True).encode()
                ).hexdigest()[:16]

                hot_spot = HotSpotDetection(
                    asset_id=asset.asset_id,
                    location=f"Measurement at {measurement.timestamp.isoformat()}",
                    severity=severity,
                    temp_differential_c=temp_diff,
                    peak_temp_c=measurement.surface_temp_c,
                    expected_temp_c=expected_surface_temp,
                    area_m2=area_m2,
                    probable_cause=probable_cause,
                    cause_confidence_percent=70.0,
                    personnel_safety_risk=personnel_risk,
                    estimated_heat_loss_w=additional_heat_loss,
                    provenance_hash=provenance_hash,
                )
                hot_spots.append(hot_spot)

        logger.info(
            f"Hot spot detection complete: asset_id={asset.asset_id}, "
            f"detected={len(hot_spots)}"
        )

        return hot_spots

    async def calculate_heat_loss(
        self,
        asset: InsulationAsset,
        measurements: Optional[List[ThermalMeasurement]] = None,
    ) -> HeatLossResult:
        """
        Calculate heat loss for an insulation asset.

        ZERO-HALLUCINATION: Uses only thermodynamic formulas.

        Heat Transfer Equations:
        - Conduction: Q = k * A * (T1 - T2) / thickness
        - Convection: Q = h * A * (Ts - Ta)
        - Radiation: Q = epsilon * sigma * A * (Ts^4 - Ta^4)

        Args:
            asset: Insulation asset definition
            measurements: Optional thermal measurements for actual surface temp

        Returns:
            HeatLossResult with calculated values
        """
        # Get measured surface temperature if available
        surface_temp_c = None
        wind_speed = 0.0

        if measurements:
            # Use average of good quality measurements
            good_temps = [
                m.surface_temp_c for m in measurements
                if m.data_quality in [DataQuality.EXCELLENT, DataQuality.GOOD, DataQuality.ACCEPTABLE]
            ]
            if good_temps:
                surface_temp_c = sum(good_temps) / len(good_temps)

            # Average wind speed
            wind_speeds = [m.wind_speed_ms for m in measurements]
            wind_speed = sum(wind_speeds) / len(wind_speeds) if wind_speeds else 0.0

        # Calculate heat loss using deterministic formula
        heat_loss = HeatLossResult.calculate(
            asset=asset,
            surface_temp_c=surface_temp_c,
            wind_speed_ms=wind_speed,
            energy_cost_usd_per_kwh=self.settings.economic.energy_cost_usd_per_kwh,
            co2_factor_kg_per_kwh=self.settings.economic.co2_emission_factor_kg_per_kwh,
            operating_hours=self.settings.economic.operating_hours_per_year,
        )

        # Calculate baseline (ideal) heat loss for comparison
        baseline = await self._get_baseline_heat_loss(asset)
        heat_loss.excess_heat_loss_w = max(0, heat_loss.heat_loss_w - baseline.heat_loss_w)
        heat_loss.excess_cost_usd_year = max(
            0, heat_loss.energy_cost_usd_year - baseline.energy_cost_usd_year
        )

        logger.info(
            f"Heat loss calculated: asset_id={asset.asset_id}, "
            f"Q={heat_loss.heat_loss_w:.1f}W, "
            f"cost=${heat_loss.energy_cost_usd_year:.0f}/year"
        )

        return heat_loss

    async def generate_repair_recommendation(
        self,
        asset: InsulationAsset,
        condition: InsulationCondition,
        heat_loss: Optional[HeatLossResult] = None,
    ) -> RepairRecommendation:
        """
        Generate repair recommendation based on condition assessment.

        ZERO-HALLUCINATION: Uses deterministic rules and calculations.

        Args:
            asset: Insulation asset
            condition: Condition assessment
            heat_loss: Heat loss calculation (calculates if None)

        Returns:
            RepairRecommendation with cost-benefit analysis
        """
        # Calculate heat loss if not provided
        if heat_loss is None:
            heat_loss = await self.calculate_heat_loss(asset)

        # Generate recommendation using deterministic formula
        recommendation = RepairRecommendation.generate(
            asset=asset,
            condition=condition,
            heat_loss=heat_loss,
            labor_rate=self.settings.economic.labor_rate_usd_per_hour,
            material_cost_per_m2=self.settings.economic.insulation_cost_usd_per_m2,
        )

        logger.info(
            f"Recommendation generated: asset_id={asset.asset_id}, "
            f"priority={recommendation.priority.value}, "
            f"repair_type={recommendation.repair_type.value}, "
            f"payback={recommendation.payback_years:.1f}yr"
        )

        return recommendation

    # =========================================================================
    # INTERNAL CALCULATION METHODS - ZERO HALLUCINATION
    # =========================================================================

    async def _assess_condition(
        self,
        asset: InsulationAsset,
        measurements: List[ThermalMeasurement],
        hot_spots: List[HotSpotDetection],
        heat_loss: HeatLossResult,
    ) -> InsulationCondition:
        """
        Assess insulation condition using deterministic formulas.

        ZERO-HALLUCINATION: Uses only physics-based calculations.
        """
        # Calculate thermal efficiency
        # Compare actual heat loss to ideal heat loss
        baseline = await self._get_baseline_heat_loss(asset)
        if baseline.heat_loss_w > 0:
            # Efficiency = 1 - (excess loss / baseline loss)
            excess_ratio = (heat_loss.heat_loss_w - baseline.heat_loss_w) / baseline.heat_loss_w
            thermal_efficiency = max(0, min(100, 100 * (1 - excess_ratio)))
        else:
            thermal_efficiency = 100.0

        # Count hot spots
        critical_hot_spots = len([
            hs for hs in hot_spots
            if hs.severity in [HotSpotSeverity.CRITICAL, HotSpotSeverity.EMERGENCY]
        ])

        # Infer damage types from hot spots
        damage_types: List[DamageType] = []
        for hs in hot_spots:
            if hs.probable_cause != DamageType.NONE and hs.probable_cause not in damage_types:
                damage_types.append(hs.probable_cause)

        # Calculate condition using deterministic method
        condition = InsulationCondition.calculate(
            asset_id=asset.asset_id,
            thermal_efficiency=thermal_efficiency,
            hot_spot_count=len(hot_spots),
            critical_hot_spots=critical_hot_spots,
            damage_types=damage_types,
            asset_age_years=asset.asset_age_years,
            typical_lifetime_years=self.settings.insulation.typical_lifetime_years,
        )

        # Add effective thickness calculation
        if condition.thermal_efficiency_percent > 0:
            condition.effective_thickness_mm = (
                asset.thickness_mm * (condition.thermal_efficiency_percent / 100.0)
            )
        else:
            condition.effective_thickness_mm = 0.0

        return condition

    def _calculate_expected_surface_temp(self, asset: InsulationAsset) -> float:
        """
        Calculate expected surface temperature for properly insulated asset.

        ZERO-HALLUCINATION: Uses thermal resistance calculation.

        For a properly insulated surface:
        T_surface = T_ambient + (T_operating - T_ambient) * (R_surface / R_total)

        where:
        R_insulation = thickness / (k * A)
        R_surface = 1 / (h * A)
        """
        # Get properties
        k = asset.effective_thermal_conductivity  # W/m-K
        t = asset.thickness_mm / 1000.0  # m

        # Thermal resistance of insulation (per unit area)
        R_insulation = t / k  # m2-K/W

        # Surface convection resistance (natural convection)
        h_conv = self.settings.thermal.natural_convection_coefficient  # W/m2-K
        R_surface = 1.0 / h_conv  # m2-K/W

        # Total resistance
        R_total = R_insulation + R_surface

        # Temperature drop across surface resistance
        delta_T = asset.operating_temp_c - asset.ambient_temp_c
        T_surface = asset.ambient_temp_c + delta_T * (R_surface / R_total)

        return T_surface

    def _infer_damage_cause(
        self,
        temp_differential: float,
        measurement: ThermalMeasurement,
        asset: InsulationAsset,
    ) -> DamageType:
        """
        Infer probable cause of thermal anomaly.

        ZERO-HALLUCINATION: Uses rule-based inference, not ML prediction.
        """
        # High humidity + high temp differential suggests moisture
        if (measurement.relative_humidity_percent > 70 and
                temp_differential > 20):
            return DamageType.MOISTURE_INGRESS

        # Very high differential with localized pattern suggests missing section
        if temp_differential > self.settings.thermal.hot_spot_emergency_delta_c:
            return DamageType.MISSING_SECTION

        # Check for age-related degradation
        if asset.asset_age_years and asset.asset_age_years > 15:
            return DamageType.THERMAL_DEGRADATION

        # Moderate differential could be compression or physical damage
        if temp_differential > self.settings.thermal.hot_spot_critical_delta_c:
            return DamageType.PHYSICAL_DAMAGE

        # Default to thermal degradation
        return DamageType.THERMAL_DEGRADATION

    async def _get_baseline_heat_loss(self, asset: InsulationAsset) -> HeatLossResult:
        """
        Get baseline (ideal) heat loss for comparison.

        Uses cached value if available, otherwise calculates.
        """
        cache_key = f"{asset.asset_id}_{asset.thickness_mm}_{asset.operating_temp_c}"

        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]

        # Calculate ideal heat loss (no degradation)
        # Create idealized version of asset
        baseline = HeatLossResult.calculate(
            asset=asset,
            surface_temp_c=None,  # Calculate ideal surface temp
            wind_speed_ms=0.0,
            energy_cost_usd_per_kwh=self.settings.economic.energy_cost_usd_per_kwh,
            co2_factor_kg_per_kwh=self.settings.economic.co2_emission_factor_kg_per_kwh,
            operating_hours=self.settings.economic.operating_hours_per_year,
        )

        self._baseline_cache[cache_key] = baseline
        return baseline

    def _filter_valid_measurements(
        self,
        measurements: List[ThermalMeasurement],
    ) -> List[ThermalMeasurement]:
        """
        Filter measurements to include only valid data.

        Args:
            measurements: All measurements

        Returns:
            Filtered list of valid measurements
        """
        valid = []
        for m in measurements:
            if m.data_quality in [
                DataQuality.EXCELLENT,
                DataQuality.GOOD,
                DataQuality.ACCEPTABLE,
            ]:
                valid.append(m)
            elif m.data_quality == DataQuality.SUSPECT:
                # Include suspect data with warning
                logger.warning(
                    f"Including suspect measurement: {m.measurement_id}"
                )
                valid.append(m)
            # Skip BAD and INTERPOLATED data

        return valid

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _log_calculation_event(
        self,
        calculation_type: str,
        asset_id: str,
        inputs_hash: str,
        outputs_hash: str,
        execution_time_ms: float,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Log calculation event for audit trail."""
        import uuid

        event = CalculationEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            calculation_type=calculation_type,
            asset_id=asset_id,
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            execution_time_ms=execution_time_ms,
            success=success,
            error_message=error_message,
        )

        self._calculation_events.append(event)

        # Trim history if needed
        if len(self._calculation_events) > self._max_event_history:
            self._calculation_events = self._calculation_events[-self._max_event_history // 2:]

    def clear_baseline_cache(self) -> None:
        """Clear cached baseline heat loss values."""
        self._baseline_cache.clear()
        logger.info("Cleared baseline cache")

    # =========================================================================
    # STATUS AND HEALTH
    # =========================================================================

    def get_status(self) -> OrchestratorStatus:
        """Get current orchestrator status."""
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        return OrchestratorStatus(
            agent_id=self.settings.agent_id,
            agent_name=self.settings.agent_name,
            version=self.VERSION,
            status="running",
            health="healthy" if self._error_count < 10 else "degraded",
            uptime_seconds=uptime,
            analyses_performed=self._analyses_count,
            analyses_successful=self._successful_count,
            hot_spots_detected=self._hot_spots_total,
            last_analysis_time=self._last_analysis_time,
            error_count=self._error_count,
        )

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Dictionary with health status and component checks
        """
        status = self.get_status()

        checks = {
            "thermal_engine": "ok",
            "hot_spot_detector": "ok" if self.settings.features.enable_hot_spot_detection else "disabled",
            "heat_loss_calculator": "ok",
            "recommendation_engine": "ok" if self.settings.features.enable_repair_recommendations else "disabled",
            "ml_service": "ok" if self.settings.features.enable_ml_predictions else "disabled",
            "thermal_imaging": "ok" if self.settings.features.enable_thermal_imaging else "disabled",
            "seed_manager": "ok",
        }

        return {
            "status": status.health,
            "version": self.VERSION,
            "uptime_seconds": status.uptime_seconds,
            "checks": checks,
            "statistics": {
                "analyses_performed": status.analyses_performed,
                "analyses_successful": status.analyses_successful,
                "success_rate": (
                    status.analyses_successful / status.analyses_performed * 100
                    if status.analyses_performed > 0 else 100.0
                ),
                "hot_spots_detected": status.hot_spots_detected,
                "error_count": status.error_count,
            },
        }

    def get_audit_trail(
        self,
        limit: int = 100,
        calculation_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get calculation audit trail.

        Args:
            limit: Maximum number of events to return
            calculation_type: Filter by calculation type

        Returns:
            List of calculation event dictionaries
        """
        events = self._calculation_events[-limit:]

        if calculation_type:
            events = [e for e in events if e.calculation_type == calculation_type]

        return [e.to_dict() for e in events]


# =============================================================================
# SYNCHRONOUS WRAPPER
# =============================================================================


def run_analysis_sync(
    asset: InsulationAsset,
    measurements: List[ThermalMeasurement],
    settings: Optional[InsulscanSettings] = None,
) -> AnalysisResult:
    """
    Run analysis synchronously.

    Convenience function for non-async contexts.

    Args:
        asset: Insulation asset definition
        measurements: Thermal measurements
        settings: Agent settings (optional)

    Returns:
        AnalysisResult with condition, heat loss, and recommendations
    """
    orchestrator = InsulscanOrchestrator(settings)
    return asyncio.run(orchestrator.analyze_insulation(asset, measurements))


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "InsulscanOrchestrator",
    "CalculationEvent",
    "OrchestratorStatus",
    "run_analysis_sync",
]
