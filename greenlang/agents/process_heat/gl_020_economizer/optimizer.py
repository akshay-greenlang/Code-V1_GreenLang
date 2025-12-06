"""
GL-020 ECONOPULSE - Main Economizer Optimizer Agent

Main agent class that orchestrates all economizer optimization components.

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units
    - Verhoff & Banchero (1974) for acid dew point

Zero-Hallucination: All calculations are deterministic with full provenance.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from .config import EconomizerOptimizationConfig
from .schemas import (
    EconomizerInput,
    EconomizerOutput,
    GasSideFoulingResult,
    WaterSideFoulingResult,
    SootBlowerResult,
    AcidDewPointResult,
    EffectivenessResult,
    SteamingResult,
    Alert,
    AlertSeverity,
    OptimizationRecommendation,
    FoulingType,
    FoulingSeverity,
    CleaningStatus,
    SootBlowingStatus,
    EconomizerStatus,
)
from .acid_dew_point import AcidDewPointCalculator, AcidDewPointInput
from .effectiveness import EffectivenessCalculator, EffectivenessInput


class EconomizerOptimizer:
    """
    GL-020 ECONOPULSE Economizer Optimization Agent.

    Provides comprehensive economizer performance optimization including:
    - Gas-side fouling detection and trending
    - Water-side scaling/fouling analysis
    - Soot blower optimization
    - Acid dew point calculations for cold-end corrosion prevention
    - Heat transfer effectiveness monitoring
    - Steaming economizer detection

    All calculations are deterministic (zero-hallucination) with full
    provenance tracking for regulatory compliance.
    """

    AGENT_ID = "GL-020"
    AGENT_NAME = "ECONOPULSE"
    VERSION = "1.0.0"

    def __init__(self, config: EconomizerOptimizationConfig):
        """
        Initialize the economizer optimizer.

        Args:
            config: Economizer optimization configuration
        """
        self.config = config
        self.economizer_id = config.economizer_id

        # Initialize calculators
        self.acid_dew_point_calc = AcidDewPointCalculator(
            safety_margin_f=config.acid_dew_point.acid_dew_point_margin_f
        )
        self.effectiveness_calc = EffectivenessCalculator()

        # State tracking
        self._last_soot_blow_time: Optional[datetime] = None
        self._effectiveness_history: List[float] = []
        self._fouling_trend_data: List[Dict] = []

    def process(self, input_data: EconomizerInput) -> EconomizerOutput:
        """
        Process economizer data and generate optimization output.

        Args:
            input_data: Current economizer measurements

        Returns:
            EconomizerOutput with complete analysis
        """
        start_time = time.time()
        alerts: List[Alert] = []
        recommendations: List[OptimizationRecommendation] = []

        # 1. Acid Dew Point Analysis
        acid_dew_point_result = self._analyze_acid_dew_point(input_data, alerts)

        # 2. Effectiveness Analysis
        effectiveness_result = self._analyze_effectiveness(input_data, alerts)

        # 3. Gas-Side Fouling Analysis
        gas_side_result = self._analyze_gas_side_fouling(input_data, effectiveness_result, alerts)

        # 4. Water-Side Fouling Analysis
        water_side_result = self._analyze_water_side_fouling(input_data, alerts)

        # 5. Soot Blower Optimization
        soot_blower_result = self._optimize_soot_blowing(
            input_data, gas_side_result, effectiveness_result, alerts
        )

        # 6. Steaming Detection
        steaming_result = self._detect_steaming(input_data, alerts)

        # 7. Determine primary fouling source
        primary_fouling_type, overall_severity = self._determine_fouling_source(
            gas_side_result, water_side_result
        )

        # 8. Generate recommendations
        recommendations = self._generate_recommendations(
            gas_side_result,
            water_side_result,
            soot_blower_result,
            acid_dew_point_result,
            effectiveness_result,
            steaming_result,
        )

        # 9. Determine overall operating status
        operating_status = self._determine_operating_status(
            gas_side_result,
            acid_dew_point_result,
            steaming_result,
            alerts,
        )

        # 10. Calculate KPIs
        kpis = self._calculate_kpis(
            effectiveness_result,
            gas_side_result,
            acid_dew_point_result,
        )

        # Build output
        processing_time_ms = (time.time() - start_time) * 1000

        output = EconomizerOutput(
            economizer_id=self.economizer_id,
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            status="success",
            operating_status=operating_status,
            processing_time_ms=processing_time_ms,
            gas_side_fouling=gas_side_result,
            water_side_fouling=water_side_result,
            soot_blower=soot_blower_result,
            acid_dew_point=acid_dew_point_result,
            effectiveness=effectiveness_result,
            steaming=steaming_result,
            primary_fouling_type=primary_fouling_type,
            overall_fouling_severity=overall_severity,
            recommendations=recommendations,
            alerts=alerts,
            kpis=kpis,
            metadata={
                "agent_id": self.AGENT_ID,
                "agent_name": self.AGENT_NAME,
                "version": self.VERSION,
                "config_id": self.config.economizer_id,
            },
        )

        # Add provenance hash
        output.input_hash = hashlib.sha256(
            json.dumps(input_data.__dict__, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        output.provenance_hash = hashlib.sha256(
            json.dumps(output.__dict__, sort_keys=True, default=str).encode()
        ).hexdigest()

        return output

    def _analyze_acid_dew_point(
        self,
        input_data: EconomizerInput,
        alerts: List[Alert],
    ) -> AcidDewPointResult:
        """Perform acid dew point analysis."""
        # Prepare input
        adp_input = AcidDewPointInput(
            flue_gas_moisture_pct=input_data.flue_gas_moisture_pct,
            flue_gas_so2_ppm=input_data.flue_gas_so2_ppm,
            fuel_sulfur_pct=input_data.fuel_sulfur_pct or self.config.acid_dew_point.fuel_sulfur_pct,
            so2_to_so3_conversion_pct=self.config.acid_dew_point.so3_conversion_pct,
            flue_gas_o2_pct=input_data.flue_gas_o2_pct,
            cold_end_metal_temp_f=input_data.cold_end_metal_temp_f or input_data.gas_outlet_temp_f,
            safety_margin_f=self.config.acid_dew_point.acid_dew_point_margin_f,
        )

        # Calculate
        result_dict = self.acid_dew_point_calc.calculate(adp_input)

        # Generate alerts
        if result_dict["corrosion_risk"] == "critical":
            alerts.append(Alert(
                severity=AlertSeverity.CRITICAL,
                category="acid_dew_point",
                title="Metal Temperature Below Acid Dew Point",
                description="Cold-end metal temperature is below the sulfuric acid dew point. Active corrosion is occurring.",
                value=result_dict["min_metal_temp_f"],
                threshold=result_dict["sulfuric_acid_dew_point_f"],
                unit="F",
                recommended_action="Increase feedwater temperature immediately.",
            ))
        elif result_dict["corrosion_risk"] == "high":
            alerts.append(Alert(
                severity=AlertSeverity.ALARM,
                category="acid_dew_point",
                title="Inadequate Acid Dew Point Margin",
                description="Metal temperature margin above acid dew point is insufficient.",
                value=result_dict["margin_above_dew_point_f"],
                threshold=self.config.acid_dew_point.acid_dew_point_margin_f,
                unit="F",
                recommended_action="Consider increasing feedwater temperature.",
            ))

        # Convert to schema
        return AcidDewPointResult(
            sulfuric_acid_dew_point_f=result_dict["sulfuric_acid_dew_point_f"],
            water_dew_point_f=result_dict["water_dew_point_f"],
            effective_dew_point_f=result_dict["effective_dew_point_f"],
            min_metal_temp_f=result_dict["min_metal_temp_f"],
            avg_metal_temp_f=result_dict["avg_metal_temp_f"],
            margin_above_dew_point_f=result_dict["margin_above_dew_point_f"],
            corrosion_risk=result_dict["corrosion_risk"],
            below_dew_point=result_dict["below_dew_point"],
            margin_adequate=result_dict["margin_adequate"],
            so3_concentration_ppm=result_dict["so3_concentration_ppm"],
            h2o_concentration_pct=result_dict["h2o_concentration_pct"],
            excess_air_pct=result_dict["excess_air_pct"],
            min_recommended_metal_temp_f=result_dict["min_recommended_metal_temp_f"],
            feedwater_temp_adjustment_f=result_dict.get("feedwater_temp_adjustment_f"),
            action_required=result_dict["action_required"],
            recommended_action=result_dict.get("recommended_action"),
            calculation_method=result_dict["calculation_method"],
            formula_reference=result_dict["formula_reference"],
        )

    def _analyze_effectiveness(
        self,
        input_data: EconomizerInput,
        alerts: List[Alert],
    ) -> EffectivenessResult:
        """Perform heat transfer effectiveness analysis."""
        # Prepare input
        eff_input = EffectivenessInput(
            gas_inlet_temp_f=input_data.gas_inlet_temp_f,
            gas_outlet_temp_f=input_data.gas_outlet_temp_f,
            water_inlet_temp_f=input_data.water_inlet_temp_f,
            water_outlet_temp_f=input_data.water_outlet_temp_f,
            gas_flow_lb_hr=input_data.gas_inlet_flow_lb_hr,
            water_flow_lb_hr=input_data.water_inlet_flow_lb_hr,
            design_effectiveness=self.config.effectiveness.design_effectiveness,
            design_ua_btu_hr_f=self.config.baseline.design_ua_btu_hr_f,
            clean_ua_btu_hr_f=self.config.baseline.clean_ua_btu_hr_f,
            design_ntu=self.config.effectiveness.design_ntu,
            flow_arrangement=self.config.design.arrangement,
        )

        # Calculate
        result_dict = self.effectiveness_calc.calculate(eff_input)

        # Track history
        self._effectiveness_history.append(result_dict["current_effectiveness"])
        if len(self._effectiveness_history) > 1000:
            self._effectiveness_history = self._effectiveness_history[-1000:]

        # Generate alerts
        eff_ratio_pct = result_dict["effectiveness_ratio"] * 100
        if eff_ratio_pct < self.config.effectiveness.effectiveness_alarm_pct:
            alerts.append(Alert(
                severity=AlertSeverity.ALARM,
                category="effectiveness",
                title="Low Heat Transfer Effectiveness",
                description="Economizer effectiveness is significantly below design value.",
                value=eff_ratio_pct,
                threshold=self.config.effectiveness.effectiveness_alarm_pct,
                unit="%",
                recommended_action="Schedule cleaning. Check for fouling on both gas and water sides.",
            ))
        elif eff_ratio_pct < self.config.effectiveness.effectiveness_warning_pct:
            alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                category="effectiveness",
                title="Degraded Heat Transfer Effectiveness",
                description="Economizer effectiveness is below warning threshold.",
                value=eff_ratio_pct,
                threshold=self.config.effectiveness.effectiveness_warning_pct,
                unit="%",
                recommended_action="Monitor closely. Consider soot blowing if gas-side fouling suspected.",
            ))

        return EffectivenessResult(
            current_effectiveness=result_dict["current_effectiveness"],
            design_effectiveness=result_dict["design_effectiveness"],
            effectiveness_ratio=result_dict["effectiveness_ratio"],
            effectiveness_deviation_pct=result_dict["effectiveness_deviation_pct"],
            current_ntu=result_dict["current_ntu"],
            design_ntu=result_dict["design_ntu"],
            current_ua_btu_hr_f=result_dict["current_ua_btu_hr_f"],
            design_ua_btu_hr_f=result_dict["design_ua_btu_hr_f"],
            clean_ua_btu_hr_f=result_dict["clean_ua_btu_hr_f"],
            ua_degradation_pct=result_dict["ua_degradation_pct"],
            actual_duty_btu_hr=result_dict["actual_duty_btu_hr"],
            expected_duty_btu_hr=result_dict["expected_duty_btu_hr"],
            duty_deficit_btu_hr=result_dict["duty_deficit_btu_hr"],
            lmtd_f=result_dict["lmtd_f"],
            approach_temp_f=result_dict["approach_temp_f"],
            gas_temp_drop_f=result_dict["gas_temp_drop_f"],
            water_temp_rise_f=result_dict["water_temp_rise_f"],
            c_min_btu_hr_f=result_dict["c_min_btu_hr_f"],
            c_max_btu_hr_f=result_dict["c_max_btu_hr_f"],
            capacity_ratio=result_dict["capacity_ratio"],
            performance_status=result_dict["performance_status"],
            primary_degradation_source=result_dict["primary_degradation_source"],
            calculation_method=result_dict["calculation_method"],
            formula_reference=result_dict["formula_reference"],
        )

    def _analyze_gas_side_fouling(
        self,
        input_data: EconomizerInput,
        effectiveness: EffectivenessResult,
        alerts: List[Alert],
    ) -> GasSideFoulingResult:
        """Analyze gas-side fouling from pressure drop and heat transfer."""
        # Get current and design pressure drops
        current_dp = input_data.gas_side_dp_in_wc or 0.0
        design_dp = self.config.baseline.design_gas_dp_in_wc

        # Flow correction (DP proportional to velocity squared)
        flow_ratio = input_data.gas_inlet_flow_lb_hr / self.config.baseline.design_gas_flow_lb_hr
        corrected_dp = current_dp / (flow_ratio ** 2) if flow_ratio > 0 else current_dp

        # DP ratio
        dp_ratio = corrected_dp / design_dp if design_dp > 0 else 1.0
        dp_deviation_pct = (dp_ratio - 1.0) * 100

        # Heat transfer degradation
        u_degradation_pct = effectiveness.ua_degradation_pct

        # Determine fouling severity
        if dp_ratio >= self.config.gas_side.dp_cleaning_trigger_ratio:
            fouling_severity = FoulingSeverity.SEVERE
            cleaning_status = CleaningStatus.URGENT
        elif dp_ratio >= self.config.gas_side.dp_alarm_ratio:
            fouling_severity = FoulingSeverity.MODERATE
            cleaning_status = CleaningStatus.REQUIRED
        elif dp_ratio >= self.config.gas_side.dp_warning_ratio:
            fouling_severity = FoulingSeverity.LIGHT
            cleaning_status = CleaningStatus.RECOMMENDED
        else:
            fouling_severity = FoulingSeverity.NONE
            cleaning_status = CleaningStatus.NOT_REQUIRED

        fouling_detected = fouling_severity != FoulingSeverity.NONE

        # Estimate fouling resistance
        fouling_resistance = 0.0
        if effectiveness.clean_ua_btu_hr_f > 0 and effectiveness.current_ua_btu_hr_f > 0:
            fouling_resistance = (1 / effectiveness.current_ua_btu_hr_f) - (1 / effectiveness.clean_ua_btu_hr_f)
            fouling_resistance = max(0.0, fouling_resistance)

        # Generate alerts
        if fouling_severity == FoulingSeverity.SEVERE:
            alerts.append(Alert(
                severity=AlertSeverity.ALARM,
                category="gas_fouling",
                title="Severe Gas-Side Fouling",
                description="Gas-side pressure drop indicates severe fouling. Cleaning required.",
                value=dp_ratio,
                threshold=self.config.gas_side.dp_cleaning_trigger_ratio,
                unit="ratio",
                recommended_action="Schedule cleaning at next opportunity.",
            ))

        return GasSideFoulingResult(
            fouling_detected=fouling_detected,
            fouling_severity=fouling_severity,
            fouling_trend="stable",
            current_dp_in_wc=current_dp,
            design_dp_in_wc=design_dp,
            corrected_dp_in_wc=corrected_dp,
            dp_ratio=dp_ratio,
            dp_deviation_pct=dp_deviation_pct,
            u_actual_btu_hr_ft2_f=effectiveness.current_ua_btu_hr_f / self.config.design.total_surface_area_ft2 if self.config.design.total_surface_area_ft2 > 0 else 0.0,
            u_clean_btu_hr_ft2_f=effectiveness.clean_ua_btu_hr_f / self.config.design.total_surface_area_ft2 if self.config.design.total_surface_area_ft2 > 0 else 0.0,
            u_degradation_pct=u_degradation_pct,
            fouling_resistance_hr_ft2_f_btu=fouling_resistance,
            efficiency_loss_pct=effectiveness.effectiveness_deviation_pct,
            cleaning_status=cleaning_status,
            soot_blow_recommended=fouling_severity in [FoulingSeverity.LIGHT, FoulingSeverity.MODERATE],
            calculation_method="ASME_PTC_4.3",
        )

    def _analyze_water_side_fouling(
        self,
        input_data: EconomizerInput,
        alerts: List[Alert],
    ) -> WaterSideFoulingResult:
        """Analyze water-side fouling from pressure drop and chemistry."""
        # Get current and design pressure drops
        current_dp = input_data.water_side_dp_psi or 0.0
        design_dp = self.config.baseline.design_water_dp_psi

        # Flow correction
        flow_ratio = input_data.water_inlet_flow_lb_hr / self.config.baseline.design_water_flow_lb_hr
        corrected_dp = current_dp / (flow_ratio ** 2) if flow_ratio > 0 else current_dp

        # DP ratio
        dp_ratio = corrected_dp / design_dp if design_dp > 0 else 1.0

        # Check chemistry compliance
        chemistry_deviations = []
        if input_data.feedwater_hardness_ppm and input_data.feedwater_hardness_ppm > self.config.water_side.max_hardness_ppm:
            chemistry_deviations.append(f"Hardness {input_data.feedwater_hardness_ppm} > {self.config.water_side.max_hardness_ppm} ppm")
        if input_data.feedwater_iron_ppm and input_data.feedwater_iron_ppm > self.config.water_side.max_iron_ppm:
            chemistry_deviations.append(f"Iron {input_data.feedwater_iron_ppm} > {self.config.water_side.max_iron_ppm} ppm")
        if input_data.feedwater_silica_ppm and input_data.feedwater_silica_ppm > self.config.water_side.max_silica_ppm:
            chemistry_deviations.append(f"Silica {input_data.feedwater_silica_ppm} > {self.config.water_side.max_silica_ppm} ppm")

        chemistry_compliant = len(chemistry_deviations) == 0

        # Determine fouling severity
        if dp_ratio >= self.config.water_side.dp_alarm_ratio:
            fouling_severity = FoulingSeverity.MODERATE
            cleaning_status = CleaningStatus.RECOMMENDED
        elif dp_ratio >= self.config.water_side.dp_warning_ratio:
            fouling_severity = FoulingSeverity.LIGHT
            cleaning_status = CleaningStatus.MONITOR
        else:
            fouling_severity = FoulingSeverity.NONE
            cleaning_status = CleaningStatus.NOT_REQUIRED

        return WaterSideFoulingResult(
            fouling_detected=fouling_severity != FoulingSeverity.NONE,
            fouling_severity=fouling_severity,
            fouling_type="scale" if chemistry_deviations else "none",
            current_dp_psi=current_dp,
            design_dp_psi=design_dp,
            corrected_dp_psi=corrected_dp,
            dp_ratio=dp_ratio,
            fouling_factor_hr_ft2_f_btu=self.config.water_side.design_fouling_factor * dp_ratio,
            design_fouling_factor=self.config.water_side.design_fouling_factor,
            fouling_factor_ratio=dp_ratio,
            chemistry_compliant=chemistry_compliant,
            chemistry_deviations=chemistry_deviations,
            cleaning_status=cleaning_status,
        )

    def _optimize_soot_blowing(
        self,
        input_data: EconomizerInput,
        gas_side: GasSideFoulingResult,
        effectiveness: EffectivenessResult,
        alerts: List[Alert],
    ) -> SootBlowerResult:
        """Optimize soot blowing schedule."""
        # Calculate hours since last blow
        hours_since_blow = 0.0
        if input_data.last_soot_blow_timestamp:
            delta = datetime.now(timezone.utc) - input_data.last_soot_blow_timestamp
            hours_since_blow = delta.total_seconds() / 3600

        # Check triggers
        dp_trigger = gas_side.dp_ratio >= self.config.soot_blower.dp_trigger_ratio
        eff_trigger = effectiveness.effectiveness_deviation_pct >= self.config.soot_blower.u_degradation_trigger_pct
        time_trigger = hours_since_blow >= self.config.soot_blower.max_interval_hours

        # Determine if blowing is recommended
        blowing_recommended = (
            (dp_trigger or eff_trigger or time_trigger)
            and hours_since_blow >= self.config.soot_blower.min_interval_hours
            and not input_data.soot_blower_active
        )

        # Calculate optimal interval based on fouling rate
        optimal_interval = self.config.soot_blower.fixed_interval_hours
        if gas_side.fouling_trend == "degrading":
            optimal_interval *= 0.8  # More frequent if degrading
        elif gas_side.fouling_trend == "improving":
            optimal_interval *= 1.2  # Less frequent if improving

        # Estimate steam consumption
        steam_per_cycle = (
            self.config.soot_blower.num_soot_blowers
            * self.config.soot_blower.steam_flow_per_blower_lb
        )

        # Determine trigger reason
        trigger_reason = ""
        if dp_trigger:
            trigger_reason = "High pressure drop"
        elif eff_trigger:
            trigger_reason = "Low heat transfer effectiveness"
        elif time_trigger:
            trigger_reason = "Maximum interval exceeded"

        return SootBlowerResult(
            blowing_recommended=blowing_recommended,
            blowing_status=SootBlowingStatus.IN_PROGRESS if input_data.soot_blower_active else SootBlowingStatus.IDLE,
            hours_since_last_blow=hours_since_blow,
            recommended_next_blow_hours=max(0, optimal_interval - hours_since_blow),
            optimal_blow_interval_hours=optimal_interval,
            dp_trigger_active=dp_trigger,
            effectiveness_trigger_active=eff_trigger,
            time_trigger_active=time_trigger,
            trigger_reason=trigger_reason,
            estimated_steam_per_cycle_lb=steam_per_cycle,
            blowing_efficiency_score=min(1.0, effectiveness.effectiveness_ratio),
        )

    def _detect_steaming(
        self,
        input_data: EconomizerInput,
        alerts: List[Alert],
    ) -> SteamingResult:
        """Detect steaming economizer conditions."""
        # Calculate saturation temperature from drum pressure
        # Using simplified correlation: T_sat(F) ≈ 212 + (P_psig / 3.5)
        saturation_temp_f = input_data.saturation_temp_f or (212 + input_data.drum_pressure_psig / 3.5)

        # Approach to saturation
        approach_temp_f = saturation_temp_f - input_data.water_outlet_temp_f
        subcooling_f = approach_temp_f

        # Water flow percentage
        water_flow_pct = (
            input_data.water_inlet_flow_lb_hr / self.config.baseline.design_water_flow_lb_hr * 100
        )

        # Low load risk
        low_load_risk = input_data.load_pct < self.config.steaming.steaming_risk_load_pct

        # Determine steaming risk
        steaming_detected = approach_temp_f <= 0
        if approach_temp_f <= self.config.steaming.approach_trip_f:
            steaming_risk = "critical"
            steaming_risk_score = 100.0
        elif approach_temp_f <= self.config.steaming.approach_alarm_f:
            steaming_risk = "high"
            steaming_risk_score = 80.0
        elif approach_temp_f <= self.config.steaming.approach_warning_f:
            steaming_risk = "moderate"
            steaming_risk_score = 50.0
        else:
            steaming_risk = "low"
            steaming_risk_score = max(0, (self.config.steaming.approach_warning_f - approach_temp_f) / self.config.steaming.approach_warning_f * 30)

        # Generate alerts
        if steaming_risk == "critical":
            alerts.append(Alert(
                severity=AlertSeverity.CRITICAL,
                category="steaming",
                title="Steaming Economizer Detected",
                description="Water outlet temperature at or above saturation. Water hammer risk.",
                value=approach_temp_f,
                threshold=self.config.steaming.approach_trip_f,
                unit="F",
                recommended_action="Increase water flow or reduce heat input immediately.",
            ))

        # Recommendations
        action_required = steaming_risk in ["critical", "high"]
        recommended_action = None
        if action_required:
            recommended_action = "Increase economizer water flow or activate recirculation."

        return SteamingResult(
            steaming_detected=steaming_detected,
            steaming_risk=steaming_risk,
            steaming_risk_score=steaming_risk_score,
            approach_temp_f=approach_temp_f,
            design_approach_f=self.config.steaming.design_approach_temp_f,
            approach_margin_f=approach_temp_f - self.config.steaming.approach_trip_f,
            water_outlet_temp_f=input_data.water_outlet_temp_f,
            saturation_temp_f=saturation_temp_f,
            subcooling_f=subcooling_f,
            current_load_pct=input_data.load_pct,
            water_flow_pct=water_flow_pct,
            low_load_risk=low_load_risk,
            action_required=action_required,
            recommended_action=recommended_action,
            increase_water_flow=action_required,
            activate_recirculation=steaming_risk == "critical" and self.config.steaming.recirculation_enabled,
            reduce_heat_input=steaming_risk == "critical",
            min_safe_load_pct=self.config.steaming.steaming_risk_load_pct,
            current_min_load_margin_pct=input_data.load_pct - self.config.steaming.steaming_risk_load_pct,
        )

    def _determine_fouling_source(
        self,
        gas_side: GasSideFoulingResult,
        water_side: WaterSideFoulingResult,
    ) -> tuple:
        """Determine primary fouling source."""
        gas_severity = gas_side.fouling_severity
        water_severity = water_side.fouling_severity

        severity_order = [
            FoulingSeverity.NONE,
            FoulingSeverity.LIGHT,
            FoulingSeverity.MODERATE,
            FoulingSeverity.SEVERE,
            FoulingSeverity.CRITICAL,
        ]

        gas_idx = severity_order.index(gas_severity)
        water_idx = severity_order.index(water_severity)

        if gas_idx > water_idx:
            return FoulingType.GAS_SIDE, gas_severity
        elif water_idx > gas_idx:
            return FoulingType.WATER_SIDE, water_severity
        elif gas_idx > 0 and water_idx > 0:
            return FoulingType.COMBINED, max(gas_severity, water_severity, key=lambda x: severity_order.index(x))
        else:
            return FoulingType.NONE, FoulingSeverity.NONE

    def _determine_operating_status(
        self,
        gas_side: GasSideFoulingResult,
        acid_dew_point: AcidDewPointResult,
        steaming: SteamingResult,
        alerts: List[Alert],
    ) -> EconomizerStatus:
        """Determine overall operating status."""
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        alarm_alerts = [a for a in alerts if a.severity == AlertSeverity.ALARM]

        if steaming.steaming_detected:
            return EconomizerStatus.STEAMING_RISK
        elif critical_alerts:
            return EconomizerStatus.TRIP
        elif alarm_alerts:
            return EconomizerStatus.ALARM
        elif gas_side.fouling_severity in [FoulingSeverity.MODERATE, FoulingSeverity.SEVERE]:
            return EconomizerStatus.DEGRADED
        else:
            return EconomizerStatus.NORMAL

    def _generate_recommendations(
        self,
        gas_side: GasSideFoulingResult,
        water_side: WaterSideFoulingResult,
        soot_blower: SootBlowerResult,
        acid_dew_point: AcidDewPointResult,
        effectiveness: EffectivenessResult,
        steaming: SteamingResult,
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        recommendations = []

        # Soot blowing recommendation
        if soot_blower.blowing_recommended:
            recommendations.append(OptimizationRecommendation(
                category="soot_blowing",
                priority=AlertSeverity.WARNING,
                title="Initiate Soot Blowing",
                description=f"Soot blowing recommended due to: {soot_blower.trigger_reason}",
                estimated_efficiency_gain_pct=effectiveness.effectiveness_deviation_pct * 0.5,
            ))

        # Acid dew point recommendation
        if acid_dew_point.action_required:
            recommendations.append(OptimizationRecommendation(
                category="acid_dew_point",
                priority=AlertSeverity.ALARM,
                title="Increase Cold-End Metal Temperature",
                description="Metal temperature is too close to acid dew point.",
                current_value=acid_dew_point.min_metal_temp_f,
                target_value=acid_dew_point.min_recommended_metal_temp_f,
                unit="F",
            ))

        # Steaming recommendation
        if steaming.action_required:
            recommendations.append(OptimizationRecommendation(
                category="steaming",
                priority=AlertSeverity.CRITICAL,
                title="Prevent Economizer Steaming",
                description="Increase water flow or reduce heat input to prevent steaming.",
                requires_outage=False,
            ))

        return recommendations

    def _calculate_kpis(
        self,
        effectiveness: EffectivenessResult,
        gas_side: GasSideFoulingResult,
        acid_dew_point: AcidDewPointResult,
    ) -> Dict[str, float]:
        """Calculate key performance indicators."""
        return {
            "effectiveness_pct": effectiveness.current_effectiveness * 100,
            "effectiveness_ratio_pct": effectiveness.effectiveness_ratio * 100,
            "gas_dp_ratio": gas_side.dp_ratio,
            "duty_mmbtu_hr": effectiveness.actual_duty_btu_hr / 1_000_000,
            "acid_dew_point_margin_f": acid_dew_point.margin_above_dew_point_f,
            "health_score": max(0, min(100, effectiveness.effectiveness_ratio * 100)),
        }


def create_economizer_optimizer(config: EconomizerOptimizationConfig) -> EconomizerOptimizer:
    """Factory function to create EconomizerOptimizer."""
    return EconomizerOptimizer(config)
