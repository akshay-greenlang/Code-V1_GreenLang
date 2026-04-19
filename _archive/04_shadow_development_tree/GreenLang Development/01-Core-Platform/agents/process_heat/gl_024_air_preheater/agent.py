"""
GL-024 AIRPREHEATER - Air Preheater Optimizer Agent

Main orchestration module for air preheater performance optimization.
Coordinates heat transfer analysis, leakage detection, cold-end protection,
fouling management, and performance optimization with full explainability.

Engineering Standards:
    - ASME PTC 4.3 - Air Heaters Performance Test Code
    - API 560 - Fired Heaters for General Refinery Service
    - NFPA 85/86 - Boiler and Combustion Systems Hazards Code

Example:
    >>> from greenlang.agents.process_heat.gl_024_air_preheater import (
    ...     AirPreheaterAgent,
    ...     AirPreheaterConfig,
    ...     AirPreheaterInput,
    ... )
    >>>
    >>> config = AirPreheaterConfig()
    >>> agent = AirPreheaterAgent(config)
    >>> result = agent.optimize(input_data)
    >>> print(f"Effectiveness: {result.heat_transfer.effectiveness_pct:.1f}%")
    >>> print(f"Leakage: {result.leakage.air_to_gas_leakage_pct:.1f}%")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

from .config import AirPreheaterConfig, AirPreheaterThresholds, OperatingMode
from .schemas import (
    AirPreheaterInput,
    AirPreheaterOutput,
    AirPreheaterType,
    HeatTransferAnalysis,
    LeakageAnalysis,
    ColdEndProtection,
    FoulingAnalysis,
    OptimizationResult,
    CorrosionRiskLevel,
    FoulingSeverity,
    AlertSeverity,
    Alert,
    Recommendation,
)
from .calculations import AirPreheaterCalculator
from .explainability import LIMEAirPreheaterExplainer, ExplainerConfig, create_explainer
from .provenance import ProvenanceTracker, ProvenanceConfig, AuditLevel

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Current state of the agent."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    GENERATING_REPORT = "generating_report"
    ERROR = "error"


@dataclass
class AgentMetrics:
    """Runtime metrics for the agent."""
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    average_processing_time_ms: float = 0.0
    last_analysis_time: Optional[datetime] = None
    alerts_generated: int = 0
    recommendations_generated: int = 0


class AirPreheaterAgent:
    """
    GL-024 Air Preheater Optimizer Agent.

    Provides comprehensive air preheater performance optimization including:
    - Heat transfer analysis (effectiveness, NTU, heat duty)
    - Leakage detection and quantification
    - Cold-end corrosion protection
    - Fouling management
    - Performance optimization with explainability

    All calculations are zero-hallucination deterministic with full
    provenance tracking for regulatory compliance.

    Attributes:
        config: Agent configuration parameters
        calculator: Zero-hallucination calculation engine
        explainer: LIME-based explainability module
        provenance: Audit trail tracker
        state: Current agent state
        metrics: Runtime performance metrics
    """

    AGENT_ID = "GL-024"
    AGENT_NAME = "AIRPREHEATER"
    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[AirPreheaterConfig] = None,
        explainer_config: Optional[ExplainerConfig] = None,
        provenance_config: Optional[ProvenanceConfig] = None,
    ):
        """
        Initialize the Air Preheater Optimizer Agent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
            explainer_config: LIME explainer configuration.
            provenance_config: Provenance tracking configuration.
        """
        self.config = config or AirPreheaterConfig()
        self.calculator = AirPreheaterCalculator(self.config)
        self.explainer = create_explainer(explainer_config)
        self.provenance = ProvenanceTracker(
            provenance_config or ProvenanceConfig(audit_level=AuditLevel.DETAILED)
        )

        self.state = AgentState.IDLE
        self.metrics = AgentMetrics()
        self._alerts: List[Alert] = []
        self._recommendations: List[Recommendation] = []

        logger.info(
            f"Initialized {self.AGENT_NAME} agent v{self.VERSION} "
            f"with config: {self.config}"
        )

    def optimize(
        self,
        input_data: AirPreheaterInput,
        include_explainability: bool = True,
        include_provenance: bool = True,
    ) -> AirPreheaterOutput:
        """
        Perform comprehensive air preheater optimization.

        This is the main entry point for the agent. It performs:
        1. Input validation
        2. Heat transfer analysis
        3. Leakage detection
        4. Cold-end protection analysis
        5. Fouling assessment
        6. Performance optimization
        7. Alert and recommendation generation
        8. Explainability report generation

        Args:
            input_data: Air preheater operating data
            include_explainability: Generate LIME explanations
            include_provenance: Track calculation provenance

        Returns:
            AirPreheaterOutput with all analysis results

        Raises:
            ValueError: If input validation fails
            RuntimeError: If calculation errors occur
        """
        start_time = datetime.now(timezone.utc)
        self.state = AgentState.ANALYZING
        self._alerts = []
        self._recommendations = []

        try:
            # Validate inputs
            self._validate_inputs(input_data)

            # Start provenance tracking
            if include_provenance:
                self.provenance.start_session(
                    equipment_tag=input_data.equipment_tag,
                    calculation_type="full_optimization",
                )

            # Perform heat transfer analysis
            heat_transfer = self._analyze_heat_transfer(input_data)

            # Perform leakage analysis
            leakage = self._analyze_leakage(input_data)

            # Perform cold-end protection analysis
            cold_end = self._analyze_cold_end_protection(input_data)

            # Perform fouling analysis
            fouling = self._analyze_fouling(input_data, heat_transfer)

            # Perform optimization
            self.state = AgentState.OPTIMIZING
            optimization = self._optimize_performance(
                input_data, heat_transfer, leakage, cold_end, fouling
            )

            # Generate alerts and recommendations
            self._generate_alerts(
                input_data, heat_transfer, leakage, cold_end, fouling
            )
            self._generate_recommendations(
                input_data, heat_transfer, leakage, cold_end, fouling, optimization
            )

            # Generate explainability reports
            self.state = AgentState.GENERATING_REPORT
            explainability_reports = {}
            if include_explainability:
                explainability_reports = self._generate_explainability(
                    input_data, heat_transfer, leakage, cold_end, fouling
                )

            # Finalize provenance
            provenance_hash = ""
            if include_provenance:
                provenance_hash = self.provenance.finalize_session()

            # Calculate processing time
            end_time = datetime.now(timezone.utc)
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            # Update metrics
            self._update_metrics(processing_time_ms, success=True)

            # Build output
            output = AirPreheaterOutput(
                timestamp=end_time,
                equipment_tag=input_data.equipment_tag,
                preheater_type=input_data.preheater_type,
                operating_mode=input_data.operating_mode,
                heat_transfer=heat_transfer,
                leakage=leakage,
                cold_end=cold_end,
                fouling=fouling,
                optimization=optimization,
                alerts=self._alerts,
                recommendations=self._recommendations,
                explainability_reports=explainability_reports,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
                agent_version=self.VERSION,
            )

            self.state = AgentState.IDLE
            logger.info(
                f"Optimization complete for {input_data.equipment_tag}: "
                f"effectiveness={heat_transfer.effectiveness_pct:.1f}%, "
                f"leakage={leakage.air_to_gas_leakage_pct:.1f}%, "
                f"cold_end_margin={cold_end.cold_end_margin_f:.1f}F"
            )

            return output

        except Exception as e:
            self.state = AgentState.ERROR
            self._update_metrics(0, success=False)
            logger.error(f"Optimization failed for {input_data.equipment_tag}: {e}")
            raise

    def analyze_heat_transfer_only(
        self,
        input_data: AirPreheaterInput,
    ) -> HeatTransferAnalysis:
        """
        Perform heat transfer analysis only.

        Lightweight method for quick effectiveness checks.

        Args:
            input_data: Air preheater operating data

        Returns:
            HeatTransferAnalysis results
        """
        self._validate_inputs(input_data)
        return self._analyze_heat_transfer(input_data)

    def check_leakage(
        self,
        input_data: AirPreheaterInput,
    ) -> LeakageAnalysis:
        """
        Perform leakage analysis only.

        Quick check for leakage levels using O2 rise method.

        Args:
            input_data: Air preheater operating data

        Returns:
            LeakageAnalysis results
        """
        self._validate_inputs(input_data)
        return self._analyze_leakage(input_data)

    def check_cold_end_status(
        self,
        input_data: AirPreheaterInput,
    ) -> ColdEndProtection:
        """
        Check cold-end corrosion protection status.

        Critical for preventing acid corrosion damage.

        Args:
            input_data: Air preheater operating data

        Returns:
            ColdEndProtection analysis
        """
        self._validate_inputs(input_data)
        return self._analyze_cold_end_protection(input_data)

    # =========================================================================
    # PRIVATE ANALYSIS METHODS
    # =========================================================================

    def _validate_inputs(self, input_data: AirPreheaterInput) -> None:
        """Validate input data for physical reasonableness."""
        # Temperature validations
        if input_data.gas_inlet_temp_f <= input_data.gas_outlet_temp_f:
            raise ValueError(
                f"Gas inlet temp ({input_data.gas_inlet_temp_f}F) must be > "
                f"outlet temp ({input_data.gas_outlet_temp_f}F)"
            )

        if input_data.air_outlet_temp_f <= input_data.air_inlet_temp_f:
            raise ValueError(
                f"Air outlet temp ({input_data.air_outlet_temp_f}F) must be > "
                f"inlet temp ({input_data.air_inlet_temp_f}F)"
            )

        # Flow rate validations
        if input_data.gas_flow_rate_lb_hr <= 0:
            raise ValueError("Gas flow rate must be positive")

        if input_data.air_flow_rate_lb_hr <= 0:
            raise ValueError("Air flow rate must be positive")

        # O2 validation
        if input_data.o2_inlet_pct < 0 or input_data.o2_inlet_pct > 21:
            raise ValueError(f"O2 inlet ({input_data.o2_inlet_pct}%) out of range 0-21%")

        logger.debug(f"Input validation passed for {input_data.equipment_tag}")

    def _analyze_heat_transfer(
        self,
        input_data: AirPreheaterInput,
    ) -> HeatTransferAnalysis:
        """Perform comprehensive heat transfer analysis."""
        # Calculate effectiveness
        effectiveness = self.calculator.calculate_effectiveness(
            gas_inlet_temp_f=input_data.gas_inlet_temp_f,
            gas_outlet_temp_f=input_data.gas_outlet_temp_f,
            air_inlet_temp_f=input_data.air_inlet_temp_f,
            air_outlet_temp_f=input_data.air_outlet_temp_f,
        )

        # Calculate NTU
        ntu = self.calculator.calculate_ntu(
            effectiveness=effectiveness.effectiveness,
            capacity_ratio=effectiveness.capacity_ratio,
            preheater_type=input_data.preheater_type,
        )

        # Calculate heat duties (both sides)
        gas_side_duty = self.calculator.calculate_heat_duty(
            flow_rate_lb_hr=input_data.gas_flow_rate_lb_hr,
            inlet_temp_f=input_data.gas_inlet_temp_f,
            outlet_temp_f=input_data.gas_outlet_temp_f,
            fluid_type="flue_gas",
            composition=input_data.gas_composition,
        )

        air_side_duty = self.calculator.calculate_heat_duty(
            flow_rate_lb_hr=input_data.air_flow_rate_lb_hr,
            inlet_temp_f=input_data.air_inlet_temp_f,
            outlet_temp_f=input_data.air_outlet_temp_f,
            fluid_type="air",
            humidity=input_data.air_humidity_pct,
        )

        # Calculate LMTD
        lmtd = self.calculator.calculate_lmtd(
            gas_inlet_temp_f=input_data.gas_inlet_temp_f,
            gas_outlet_temp_f=input_data.gas_outlet_temp_f,
            air_inlet_temp_f=input_data.air_inlet_temp_f,
            air_outlet_temp_f=input_data.air_outlet_temp_f,
        )

        # Calculate X-ratio for regenerative preheaters
        x_ratio = None
        if input_data.preheater_type == AirPreheaterType.REGENERATIVE:
            x_ratio = self.calculator.calculate_x_ratio(
                gas_inlet_temp_f=input_data.gas_inlet_temp_f,
                gas_outlet_temp_f=input_data.gas_outlet_temp_f,
                air_inlet_temp_f=input_data.air_inlet_temp_f,
                air_outlet_temp_f=input_data.air_outlet_temp_f,
                o2_inlet_pct=input_data.o2_inlet_pct,
                o2_outlet_pct=input_data.o2_outlet_pct,
            )

        # Calculate heat balance error
        heat_balance_error_pct = abs(
            (gas_side_duty.heat_duty_mmbtu_hr - air_side_duty.heat_duty_mmbtu_hr) /
            gas_side_duty.heat_duty_mmbtu_hr * 100
        ) if gas_side_duty.heat_duty_mmbtu_hr > 0 else 0.0

        # Record provenance
        self.provenance.record_calculation(
            calculation_type="heat_transfer_analysis",
            inputs={
                "gas_inlet_temp_f": input_data.gas_inlet_temp_f,
                "gas_outlet_temp_f": input_data.gas_outlet_temp_f,
                "air_inlet_temp_f": input_data.air_inlet_temp_f,
                "air_outlet_temp_f": input_data.air_outlet_temp_f,
            },
            outputs={
                "effectiveness_pct": effectiveness.effectiveness * 100,
                "ntu": ntu.ntu,
                "heat_duty_mmbtu_hr": gas_side_duty.heat_duty_mmbtu_hr,
            },
            methodology="epsilon-NTU per ASME PTC 4.3",
        )

        return HeatTransferAnalysis(
            effectiveness_pct=effectiveness.effectiveness * 100,
            effectiveness_design_pct=input_data.design_effectiveness_pct or 70.0,
            ntu=ntu.ntu,
            capacity_ratio=effectiveness.capacity_ratio,
            gas_side_heat_duty_mmbtu_hr=gas_side_duty.heat_duty_mmbtu_hr,
            air_side_heat_duty_mmbtu_hr=air_side_duty.heat_duty_mmbtu_hr,
            heat_balance_error_pct=heat_balance_error_pct,
            lmtd_f=lmtd.lmtd_f,
            ua_btu_hr_f=gas_side_duty.heat_duty_mmbtu_hr * 1e6 / lmtd.lmtd_f if lmtd.lmtd_f > 0 else 0,
            x_ratio=x_ratio.x_ratio if x_ratio else None,
            gas_temp_drop_f=input_data.gas_inlet_temp_f - input_data.gas_outlet_temp_f,
            air_temp_rise_f=input_data.air_outlet_temp_f - input_data.air_inlet_temp_f,
        )

    def _analyze_leakage(
        self,
        input_data: AirPreheaterInput,
    ) -> LeakageAnalysis:
        """Perform leakage detection and quantification."""
        # Calculate leakage using O2 rise method
        leakage = self.calculator.calculate_leakage_o2_method(
            o2_inlet_pct=input_data.o2_inlet_pct,
            o2_outlet_pct=input_data.o2_outlet_pct,
            gas_flow_rate_lb_hr=input_data.gas_flow_rate_lb_hr,
            air_flow_rate_lb_hr=input_data.air_flow_rate_lb_hr,
        )

        # Calculate seal leakage if data available
        seal_leakage = None
        if input_data.seal_clearance_in is not None:
            seal_leakage = self.calculator.calculate_seal_leakage(
                seal_clearance_in=input_data.seal_clearance_in,
                seal_diameter_in=input_data.seal_diameter_in or 300,
                pressure_differential_in_wc=input_data.pressure_differential_in_wc or 5,
                rotor_speed_rpm=input_data.rotor_speed_rpm,
            )

        # Determine leakage status
        thresholds = self.config.thresholds
        if leakage.air_to_gas_leakage_pct >= thresholds.leakage_alarm_pct:
            leakage_status = "ALARM"
        elif leakage.air_to_gas_leakage_pct >= thresholds.leakage_warning_pct:
            leakage_status = "WARNING"
        else:
            leakage_status = "NORMAL"

        # Record provenance
        self.provenance.record_calculation(
            calculation_type="leakage_analysis",
            inputs={
                "o2_inlet_pct": input_data.o2_inlet_pct,
                "o2_outlet_pct": input_data.o2_outlet_pct,
            },
            outputs={
                "air_to_gas_leakage_pct": leakage.air_to_gas_leakage_pct,
                "o2_rise_pct": leakage.o2_rise_pct,
            },
            methodology="O2 rise method per ASME PTC 4.3",
        )

        return LeakageAnalysis(
            air_to_gas_leakage_pct=leakage.air_to_gas_leakage_pct,
            gas_to_air_leakage_pct=leakage.gas_to_air_leakage_pct,
            o2_rise_pct=leakage.o2_rise_pct,
            leakage_flow_lb_hr=leakage.leakage_flow_lb_hr,
            seal_leakage_pct=seal_leakage.seal_leakage_pct if seal_leakage else None,
            seal_condition=seal_leakage.seal_condition if seal_leakage else None,
            leakage_status=leakage_status,
            efficiency_impact_pct=leakage.efficiency_impact_pct,
            corrective_action_required=leakage_status in ["WARNING", "ALARM"],
        )

    def _analyze_cold_end_protection(
        self,
        input_data: AirPreheaterInput,
    ) -> ColdEndProtection:
        """Analyze cold-end corrosion protection status."""
        # Calculate acid dew point using multiple methods
        adp_verhoff = self.calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_vol_pct=input_data.gas_composition.get("H2O", 8.0),
            so3_ppm=input_data.so3_ppm or 5.0,
        )

        adp_okkes = self.calculator.calculate_acid_dew_point_okkes(
            so2_ppm=input_data.so2_ppm or 500,
            h2o_vol_pct=input_data.gas_composition.get("H2O", 8.0),
            excess_air_pct=self._calculate_excess_air(input_data.o2_inlet_pct),
        )

        # Calculate water dew point
        water_dp = self.calculator.calculate_water_dew_point(
            h2o_vol_pct=input_data.gas_composition.get("H2O", 8.0),
            pressure_psia=input_data.gas_pressure_psia or 14.7,
        )

        # Use most conservative (highest) acid dew point
        acid_dew_point_f = max(adp_verhoff.acid_dew_point_f, adp_okkes.acid_dew_point_f)

        # Calculate cold-end element temperature
        cold_end_temp_f = self.calculator.calculate_cold_end_temp(
            gas_outlet_temp_f=input_data.gas_outlet_temp_f,
            air_inlet_temp_f=input_data.air_inlet_temp_f,
            preheater_type=input_data.preheater_type,
        )

        # Calculate margin
        cold_end_margin_f = cold_end_temp_f - acid_dew_point_f

        # Determine corrosion risk
        thresholds = self.config.thresholds
        if cold_end_margin_f <= thresholds.cold_end_margin_alarm_f:
            corrosion_risk = CorrosionRiskLevel.CRITICAL
        elif cold_end_margin_f <= thresholds.cold_end_margin_warning_f:
            corrosion_risk = CorrosionRiskLevel.HIGH
        elif cold_end_margin_f <= 30:
            corrosion_risk = CorrosionRiskLevel.MODERATE
        else:
            corrosion_risk = CorrosionRiskLevel.LOW

        # Record provenance
        self.provenance.record_calculation(
            calculation_type="cold_end_protection",
            inputs={
                "so3_ppm": input_data.so3_ppm,
                "h2o_vol_pct": input_data.gas_composition.get("H2O", 8.0),
                "gas_outlet_temp_f": input_data.gas_outlet_temp_f,
            },
            outputs={
                "acid_dew_point_f": acid_dew_point_f,
                "cold_end_margin_f": cold_end_margin_f,
                "corrosion_risk": corrosion_risk.value,
            },
            methodology="Verhoff-Banchero and Okkes correlations",
        )

        return ColdEndProtection(
            acid_dew_point_verhoff_f=adp_verhoff.acid_dew_point_f,
            acid_dew_point_okkes_f=adp_okkes.acid_dew_point_f,
            acid_dew_point_selected_f=acid_dew_point_f,
            water_dew_point_f=water_dp.water_dew_point_f,
            cold_end_element_temp_f=cold_end_temp_f,
            cold_end_margin_f=cold_end_margin_f,
            corrosion_risk=corrosion_risk,
            minimum_gas_outlet_temp_f=acid_dew_point_f + thresholds.cold_end_margin_warning_f,
            scaph_active=input_data.scaph_active or False,
            scaph_steam_flow_lb_hr=input_data.scaph_steam_flow_lb_hr,
            bisector_position_pct=input_data.bisector_position_pct,
        )

    def _analyze_fouling(
        self,
        input_data: AirPreheaterInput,
        heat_transfer: HeatTransferAnalysis,
    ) -> FoulingAnalysis:
        """Analyze fouling condition and cleaning requirements."""
        # Calculate cleanliness factor
        cleanliness = self.calculator.calculate_cleanliness_factor(
            current_effectiveness=heat_transfer.effectiveness_pct / 100,
            design_effectiveness=heat_transfer.effectiveness_design_pct / 100,
            current_ua=heat_transfer.ua_btu_hr_f,
            design_ua=input_data.design_ua_btu_hr_f,
        )

        # Calculate pressure drop ratio
        dp_ratio_gas = (
            input_data.gas_pressure_drop_in_wc / input_data.design_gas_dp_in_wc
            if input_data.design_gas_dp_in_wc and input_data.design_gas_dp_in_wc > 0
            else 1.0
        )
        dp_ratio_air = (
            input_data.air_pressure_drop_in_wc / input_data.design_air_dp_in_wc
            if input_data.design_air_dp_in_wc and input_data.design_air_dp_in_wc > 0
            else 1.0
        )

        # Determine fouling severity
        thresholds = self.config.thresholds
        if cleanliness.cleanliness_factor < 0.7 or max(dp_ratio_gas, dp_ratio_air) > thresholds.dp_ratio_alarm:
            fouling_severity = FoulingSeverity.SEVERE
        elif cleanliness.cleanliness_factor < 0.85 or max(dp_ratio_gas, dp_ratio_air) > thresholds.dp_ratio_warning:
            fouling_severity = FoulingSeverity.MODERATE
        elif cleanliness.cleanliness_factor < 0.95:
            fouling_severity = FoulingSeverity.LIGHT
        else:
            fouling_severity = FoulingSeverity.CLEAN

        # Estimate cleaning effectiveness from history if available
        last_cleaning_effectiveness = None
        if input_data.last_cleaning_date and input_data.effectiveness_after_cleaning_pct:
            last_cleaning_effectiveness = (
                input_data.effectiveness_after_cleaning_pct -
                heat_transfer.effectiveness_pct
            )

        # Calculate days since last cleaning
        days_since_cleaning = None
        if input_data.last_cleaning_date:
            days_since_cleaning = (
                datetime.now(timezone.utc) - input_data.last_cleaning_date
            ).days

        # Record provenance
        self.provenance.record_calculation(
            calculation_type="fouling_analysis",
            inputs={
                "current_effectiveness": heat_transfer.effectiveness_pct,
                "design_effectiveness": heat_transfer.effectiveness_design_pct,
                "gas_dp_in_wc": input_data.gas_pressure_drop_in_wc,
            },
            outputs={
                "cleanliness_factor": cleanliness.cleanliness_factor,
                "fouling_severity": fouling_severity.value,
            },
            methodology="UA degradation and pressure drop analysis",
        )

        return FoulingAnalysis(
            cleanliness_factor=cleanliness.cleanliness_factor,
            fouling_resistance_hr_ft2_f_btu=cleanliness.fouling_resistance,
            gas_side_dp_ratio=dp_ratio_gas,
            air_side_dp_ratio=dp_ratio_air,
            fouling_severity=fouling_severity,
            effectiveness_loss_pct=heat_transfer.effectiveness_design_pct - heat_transfer.effectiveness_pct,
            days_since_last_cleaning=days_since_cleaning,
            cleaning_recommended=fouling_severity in [FoulingSeverity.MODERATE, FoulingSeverity.SEVERE],
            estimated_cleaning_benefit_pct=cleanliness.estimated_recovery_pct,
            soot_blowing_effective=input_data.soot_blower_status == "EFFECTIVE",
        )

    def _optimize_performance(
        self,
        input_data: AirPreheaterInput,
        heat_transfer: HeatTransferAnalysis,
        leakage: LeakageAnalysis,
        cold_end: ColdEndProtection,
        fouling: FoulingAnalysis,
    ) -> OptimizationResult:
        """Optimize air preheater operation."""
        # Calculate optimal air outlet temperature
        # Balance: higher temp = better efficiency, lower temp = better cold-end protection
        optimal_air_outlet = self.calculator.calculate_optimal_air_outlet_temp(
            current_air_outlet_f=input_data.air_outlet_temp_f,
            acid_dew_point_f=cold_end.acid_dew_point_selected_f,
            cold_end_margin_target_f=self.config.thresholds.cold_end_margin_warning_f + 10,
            gas_inlet_temp_f=input_data.gas_inlet_temp_f,
            current_effectiveness=heat_transfer.effectiveness_pct / 100,
            max_effectiveness=0.85,  # Typical maximum for regenerative
        )

        # Calculate energy savings potential
        energy_savings = self.calculator.calculate_energy_savings(
            current_effectiveness=heat_transfer.effectiveness_pct / 100,
            achievable_effectiveness=min(
                heat_transfer.effectiveness_design_pct / 100,
                (heat_transfer.effectiveness_pct + fouling.estimated_cleaning_benefit_pct) / 100
            ),
            fuel_flow_mmbtu_hr=input_data.fuel_input_mmbtu_hr or 100,
            fuel_cost_per_mmbtu=input_data.fuel_cost_per_mmbtu or 5.0,
        )

        # Calculate efficiency impact on boiler
        boiler_efficiency_impact = self.calculator.calculate_efficiency_impact(
            air_temp_rise_f=heat_transfer.air_temp_rise_f,
            baseline_air_temp_rise_f=input_data.design_air_temp_rise_f or 200,
            leakage_pct=leakage.air_to_gas_leakage_pct,
        )

        # Determine optimal setpoints
        optimal_setpoints = {
            "air_outlet_temp_f": optimal_air_outlet.optimal_temp_f,
            "gas_outlet_temp_target_f": cold_end.minimum_gas_outlet_temp_f,
            "bisector_position_pct": self._calculate_optimal_bisector(
                cold_end.corrosion_risk, input_data.bisector_position_pct
            ),
        }

        # Add SCAPH control if applicable
        if cold_end.scaph_active or cold_end.corrosion_risk in [CorrosionRiskLevel.HIGH, CorrosionRiskLevel.CRITICAL]:
            optimal_setpoints["scaph_steam_flow_lb_hr"] = self._calculate_scaph_steam(
                cold_end.cold_end_margin_f,
                cold_end.corrosion_risk,
            )

        return OptimizationResult(
            optimal_setpoints=optimal_setpoints,
            current_efficiency_impact_pct=boiler_efficiency_impact.efficiency_impact_pct,
            achievable_efficiency_gain_pct=energy_savings.efficiency_gain_pct,
            annual_energy_savings_mmbtu=energy_savings.annual_savings_mmbtu,
            annual_cost_savings_usd=energy_savings.annual_cost_savings_usd,
            payback_period_months=energy_savings.payback_months,
            optimization_actions=[
                action for action in [
                    "Adjust bisector position" if cold_end.corrosion_risk != CorrosionRiskLevel.LOW else None,
                    "Schedule cleaning" if fouling.cleaning_recommended else None,
                    "Reduce leakage" if leakage.corrective_action_required else None,
                    "Activate/adjust SCAPH" if cold_end.corrosion_risk in [CorrosionRiskLevel.HIGH, CorrosionRiskLevel.CRITICAL] else None,
                ] if action
            ],
            constraints_binding=[
                constraint for constraint in [
                    "Cold-end protection" if cold_end.corrosion_risk != CorrosionRiskLevel.LOW else None,
                    "Maximum pressure drop" if fouling.gas_side_dp_ratio > 1.3 else None,
                    "Leakage limit" if leakage.air_to_gas_leakage_pct > 10 else None,
                ] if constraint
            ],
        )

    def _generate_alerts(
        self,
        input_data: AirPreheaterInput,
        heat_transfer: HeatTransferAnalysis,
        leakage: LeakageAnalysis,
        cold_end: ColdEndProtection,
        fouling: FoulingAnalysis,
    ) -> None:
        """Generate alerts based on analysis results."""
        thresholds = self.config.thresholds

        # Cold-end corrosion alerts (highest priority)
        if cold_end.corrosion_risk == CorrosionRiskLevel.CRITICAL:
            self._alerts.append(Alert(
                severity=AlertSeverity.ALARM,
                category="COLD_END_PROTECTION",
                message=f"CRITICAL: Cold-end margin only {cold_end.cold_end_margin_f:.0f}F - immediate action required",
                parameter="cold_end_margin_f",
                current_value=cold_end.cold_end_margin_f,
                threshold_value=thresholds.cold_end_margin_alarm_f,
                recommended_action="Increase gas outlet temperature or activate SCAPH immediately",
            ))
        elif cold_end.corrosion_risk == CorrosionRiskLevel.HIGH:
            self._alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                category="COLD_END_PROTECTION",
                message=f"WARNING: Cold-end margin {cold_end.cold_end_margin_f:.0f}F below recommended minimum",
                parameter="cold_end_margin_f",
                current_value=cold_end.cold_end_margin_f,
                threshold_value=thresholds.cold_end_margin_warning_f,
                recommended_action="Monitor closely and consider SCAPH activation",
            ))

        # Leakage alerts
        if leakage.leakage_status == "ALARM":
            self._alerts.append(Alert(
                severity=AlertSeverity.ALARM,
                category="LEAKAGE",
                message=f"ALARM: Air-to-gas leakage {leakage.air_to_gas_leakage_pct:.1f}% exceeds limit",
                parameter="air_to_gas_leakage_pct",
                current_value=leakage.air_to_gas_leakage_pct,
                threshold_value=thresholds.leakage_alarm_pct,
                recommended_action="Schedule seal inspection and replacement",
            ))
        elif leakage.leakage_status == "WARNING":
            self._alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                category="LEAKAGE",
                message=f"WARNING: Air-to-gas leakage {leakage.air_to_gas_leakage_pct:.1f}% elevated",
                parameter="air_to_gas_leakage_pct",
                current_value=leakage.air_to_gas_leakage_pct,
                threshold_value=thresholds.leakage_warning_pct,
                recommended_action="Plan seal inspection during next outage",
            ))

        # Fouling alerts
        if fouling.fouling_severity == FoulingSeverity.SEVERE:
            self._alerts.append(Alert(
                severity=AlertSeverity.ALARM,
                category="FOULING",
                message=f"ALARM: Severe fouling detected - cleanliness factor {fouling.cleanliness_factor:.2f}",
                parameter="cleanliness_factor",
                current_value=fouling.cleanliness_factor,
                threshold_value=0.7,
                recommended_action="Schedule immediate cleaning",
            ))
        elif fouling.fouling_severity == FoulingSeverity.MODERATE:
            self._alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                category="FOULING",
                message=f"WARNING: Moderate fouling - effectiveness loss {fouling.effectiveness_loss_pct:.1f}%",
                parameter="cleanliness_factor",
                current_value=fouling.cleanliness_factor,
                threshold_value=0.85,
                recommended_action="Plan cleaning during next scheduled outage",
            ))

        # Effectiveness degradation alert
        effectiveness_degradation = heat_transfer.effectiveness_design_pct - heat_transfer.effectiveness_pct
        if effectiveness_degradation > thresholds.effectiveness_degradation_alarm_pct:
            self._alerts.append(Alert(
                severity=AlertSeverity.ALARM,
                category="PERFORMANCE",
                message=f"ALARM: Effectiveness {effectiveness_degradation:.1f}% below design",
                parameter="effectiveness_pct",
                current_value=heat_transfer.effectiveness_pct,
                threshold_value=heat_transfer.effectiveness_design_pct - thresholds.effectiveness_degradation_alarm_pct,
                recommended_action="Investigate root cause - check leakage, fouling, and operating conditions",
            ))

        self.metrics.alerts_generated += len(self._alerts)

    def _generate_recommendations(
        self,
        input_data: AirPreheaterInput,
        heat_transfer: HeatTransferAnalysis,
        leakage: LeakageAnalysis,
        cold_end: ColdEndProtection,
        fouling: FoulingAnalysis,
        optimization: OptimizationResult,
    ) -> None:
        """Generate actionable recommendations."""
        # Optimization recommendations
        if optimization.achievable_efficiency_gain_pct > 0.5:
            self._recommendations.append(Recommendation(
                priority=1,
                category="OPTIMIZATION",
                title="Efficiency Improvement Available",
                description=(
                    f"Potential to improve boiler efficiency by {optimization.achievable_efficiency_gain_pct:.2f}% "
                    f"through air preheater optimization"
                ),
                estimated_savings_usd=optimization.annual_cost_savings_usd,
                implementation_effort="LOW",
                actions=optimization.optimization_actions,
            ))

        # Cleaning recommendations
        if fouling.cleaning_recommended:
            self._recommendations.append(Recommendation(
                priority=2 if fouling.fouling_severity == FoulingSeverity.SEVERE else 3,
                category="MAINTENANCE",
                title="Air Preheater Cleaning Recommended",
                description=(
                    f"Cleaning can recover approximately {fouling.estimated_cleaning_benefit_pct:.1f}% effectiveness. "
                    f"Current cleanliness factor: {fouling.cleanliness_factor:.2f}"
                ),
                estimated_savings_usd=optimization.annual_cost_savings_usd * 0.5,
                implementation_effort="MEDIUM",
                actions=["Schedule water wash", "Verify soot blower operation", "Check for air heater fires"],
            ))

        # Leakage reduction recommendations
        if leakage.corrective_action_required:
            self._recommendations.append(Recommendation(
                priority=2,
                category="MAINTENANCE",
                title="Seal Maintenance Required",
                description=(
                    f"Leakage at {leakage.air_to_gas_leakage_pct:.1f}% is impacting efficiency "
                    f"by approximately {leakage.efficiency_impact_pct:.2f}%"
                ),
                estimated_savings_usd=leakage.efficiency_impact_pct * input_data.fuel_input_mmbtu_hr * 8760 * 0.5 if input_data.fuel_input_mmbtu_hr else 0,
                implementation_effort="HIGH",
                actions=[
                    "Inspect radial and axial seals",
                    "Measure seal clearances",
                    "Replace worn seal segments",
                    "Verify rotor concentricity",
                ],
            ))

        # Cold-end protection recommendations
        if cold_end.corrosion_risk in [CorrosionRiskLevel.HIGH, CorrosionRiskLevel.CRITICAL]:
            self._recommendations.append(Recommendation(
                priority=1,
                category="OPERATIONS",
                title="Cold-End Protection Action Required",
                description=(
                    f"Cold-end margin of {cold_end.cold_end_margin_f:.0f}F risks acid corrosion. "
                    f"Acid dew point: {cold_end.acid_dew_point_selected_f:.0f}F"
                ),
                estimated_savings_usd=0,  # Avoided damage cost
                implementation_effort="LOW",
                actions=[
                    f"Increase gas outlet temperature to minimum {cold_end.minimum_gas_outlet_temp_f:.0f}F",
                    "Activate or increase SCAPH steam flow" if not cold_end.scaph_active else "Verify SCAPH effectiveness",
                    f"Adjust bisector to {optimization.optimal_setpoints.get('bisector_position_pct', 50):.0f}%",
                ],
            ))

        self.metrics.recommendations_generated += len(self._recommendations)

    def _generate_explainability(
        self,
        input_data: AirPreheaterInput,
        heat_transfer: HeatTransferAnalysis,
        leakage: LeakageAnalysis,
        cold_end: ColdEndProtection,
        fouling: FoulingAnalysis,
    ) -> Dict[str, Any]:
        """Generate LIME explainability reports."""
        reports = {}

        # Heat transfer explanation
        heat_features = {
            "gas_inlet_temp_f": input_data.gas_inlet_temp_f,
            "gas_outlet_temp_f": input_data.gas_outlet_temp_f,
            "air_inlet_temp_f": input_data.air_inlet_temp_f,
            "air_outlet_temp_f": input_data.air_outlet_temp_f,
            "gas_flow_rate_lb_hr": input_data.gas_flow_rate_lb_hr,
            "air_flow_rate_lb_hr": input_data.air_flow_rate_lb_hr,
        }
        reports["heat_transfer"] = self.explainer.explain_heat_transfer_performance(
            features=heat_features,
            prediction=heat_transfer.effectiveness_pct,
        )

        # Leakage explanation
        leakage_features = {
            "o2_inlet_pct": input_data.o2_inlet_pct,
            "o2_outlet_pct": input_data.o2_outlet_pct,
            "seal_clearance_in": input_data.seal_clearance_in or 0.125,
            "pressure_differential_in_wc": input_data.pressure_differential_in_wc or 5,
        }
        reports["leakage"] = self.explainer.explain_leakage_analysis(
            features=leakage_features,
            prediction=leakage.air_to_gas_leakage_pct,
        )

        # Cold-end explanation
        cold_end_features = {
            "so3_ppm": input_data.so3_ppm or 5,
            "h2o_vol_pct": input_data.gas_composition.get("H2O", 8.0),
            "gas_outlet_temp_f": input_data.gas_outlet_temp_f,
            "air_inlet_temp_f": input_data.air_inlet_temp_f,
        }
        reports["cold_end"] = self.explainer.explain_cold_end_protection(
            features=cold_end_features,
            prediction=cold_end.cold_end_margin_f,
        )

        return reports

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _calculate_excess_air(self, o2_pct: float) -> float:
        """Calculate excess air percentage from O2."""
        # Approximate formula: EA% = O2% / (21 - O2%) * 100
        if o2_pct >= 21:
            return 100.0
        return o2_pct / (21 - o2_pct) * 100

    def _calculate_optimal_bisector(
        self,
        corrosion_risk: CorrosionRiskLevel,
        current_position: Optional[float],
    ) -> float:
        """Calculate optimal bisector position for cold-end protection."""
        # Bisector directs more gas to cold end to raise temperatures
        if corrosion_risk == CorrosionRiskLevel.CRITICAL:
            return 70.0  # Maximum cold-end protection
        elif corrosion_risk == CorrosionRiskLevel.HIGH:
            return 60.0
        elif corrosion_risk == CorrosionRiskLevel.MODERATE:
            return 55.0
        else:
            return current_position or 50.0  # Neutral position

    def _calculate_scaph_steam(
        self,
        cold_end_margin: float,
        corrosion_risk: CorrosionRiskLevel,
    ) -> float:
        """Calculate required SCAPH steam flow."""
        # Approximate: need more steam as margin decreases
        if corrosion_risk == CorrosionRiskLevel.CRITICAL:
            return 5000.0  # Maximum steam flow
        elif corrosion_risk == CorrosionRiskLevel.HIGH:
            return 3000.0
        elif cold_end_margin < 30:
            return 2000.0
        else:
            return 0.0  # Not required

    def _update_metrics(self, processing_time_ms: float, success: bool) -> None:
        """Update agent runtime metrics."""
        self.metrics.total_analyses += 1
        if success:
            self.metrics.successful_analyses += 1
        else:
            self.metrics.failed_analyses += 1

        # Update rolling average
        n = self.metrics.successful_analyses
        if n > 0:
            self.metrics.average_processing_time_ms = (
                (self.metrics.average_processing_time_ms * (n - 1) + processing_time_ms) / n
            )

        self.metrics.last_analysis_time = datetime.now(timezone.utc)

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "state": self.state.value,
            "metrics": {
                "total_analyses": self.metrics.total_analyses,
                "successful_analyses": self.metrics.successful_analyses,
                "failed_analyses": self.metrics.failed_analyses,
                "success_rate_pct": (
                    self.metrics.successful_analyses / self.metrics.total_analyses * 100
                    if self.metrics.total_analyses > 0 else 0
                ),
                "average_processing_time_ms": self.metrics.average_processing_time_ms,
                "last_analysis_time": (
                    self.metrics.last_analysis_time.isoformat()
                    if self.metrics.last_analysis_time else None
                ),
                "alerts_generated": self.metrics.alerts_generated,
                "recommendations_generated": self.metrics.recommendations_generated,
            },
        }


def create_agent(
    config: Optional[AirPreheaterConfig] = None,
    explainer_config: Optional[ExplainerConfig] = None,
    provenance_config: Optional[ProvenanceConfig] = None,
) -> AirPreheaterAgent:
    """
    Factory function to create an Air Preheater Optimizer Agent.

    Args:
        config: Agent configuration. Uses defaults if not provided.
        explainer_config: LIME explainer configuration.
        provenance_config: Provenance tracking configuration.

    Returns:
        Configured AirPreheaterAgent instance
    """
    return AirPreheaterAgent(
        config=config,
        explainer_config=explainer_config,
        provenance_config=provenance_config,
    )
