"""
CondenserOptimizationAgent - HEI Standards compliant condenser optimization

This module implements the CondenserOptimizationAgent (GL-017 CONDENSYNC)
for optimizing steam surface condenser performance using HEI Standards,
fouling prediction, and SHAP/LIME explainability.

The agent follows GreenLang's zero-hallucination principle by using only
deterministic calculations from HEI Standards and heat transfer engineering -
no ML/LLM in the calculation path.

Example:
    >>> config = AgentConfig(agent_id="GL-017")
    >>> agent = CondenserOptimizationAgent(config)
    >>> result = agent.run(input_data)
    >>> assert result.validation_status == "PASS"
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import logging

from .schemas import (
    CondenserInput,
    CondenserOutput,
    AgentConfig,
    HeatTransferAnalysis,
    CleanlinessAnalysis,
    VacuumAnalysis,
    FoulingAnalysis,
    AirLeakageAnalysis,
    OptimizationRecommendation,
    CleaningSchedule,
    ExplainabilityReport,
    OptimizationPriority,
    CleaningMethod,
)

from .calculators import (
    # Heat transfer
    calculate_overall_heat_transfer_coefficient,
    calculate_lmtd,
    calculate_heat_duty,
    calculate_ttd,
    calculate_cooling_water_rise,
    estimate_latent_heat,
    HEI_MATERIAL_FACTORS,
    # Vacuum
    calculate_saturation_pressure,
    calculate_saturation_temperature,
    calculate_vacuum_inches_hg,
    calculate_vacuum_efficiency,
    calculate_theoretical_vacuum,
    calculate_vacuum_deviation,
    calculate_power_loss_from_vacuum_degradation,
    assess_air_leakage_severity,
    generate_vacuum_recommendations,
    # Fouling
    calculate_fouling_resistance,
    calculate_fouling_rate,
    predict_fouling_linear,
    calculate_hours_to_cleaning_threshold,
    recommend_cleaning_date,
    assess_fouling_severity,
    get_standard_fouling_resistance,
    generate_fouling_recommendations,
    estimate_fouling_parameters_from_history,
    CLEANING_EFFECTIVENESS,
    # Cleanliness
    calculate_cleanliness_factor,
    assess_cleanliness_status,
    calculate_cf_trend,
    calculate_performance_impact,
    calculate_cleaning_benefit_cf,
    get_design_cleanliness_factor,
    generate_cf_recommendations,
    calculate_cf_score,
)

logger = logging.getLogger(__name__)


class CondenserOptimizationAgent:
    """
    CondenserOptimizationAgent implementation (GL-017 CONDENSYNC).

    This agent performs comprehensive condenser performance optimization
    following HEI Standards for Steam Surface Condensers. It provides:
    - Heat transfer analysis with HEI correlations
    - Cleanliness factor tracking and trending
    - Vacuum system optimization
    - Fouling prediction and cleaning scheduling
    - SHAP/LIME-style explainability reports

    Zero-hallucination approach:
    - All heat transfer calculations use HEI Standards formulas
    - Fouling predictions use statistical regression (not ML inference)
    - All outputs are bounded by physical constraints

    Attributes:
        config: Agent configuration
        agent_id: Unique agent identifier (GL-017)
        agent_name: Human-readable agent name (CONDENSYNC)
        version: Agent version string

    Example:
        >>> config = AgentConfig()
        >>> agent = CondenserOptimizationAgent(config)
        >>> input_data = CondenserInput(
        ...     condenser_id="COND-001",
        ...     design=CondenserDesignData(...),
        ...     cooling_water=CoolingWaterData(...),
        ...     vacuum=VacuumData(...)
        ... )
        >>> result = agent.run(input_data)
        >>> assert result.validation_status == "PASS"
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize CondenserOptimizationAgent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        self.config = config or AgentConfig()
        self.agent_id = self.config.agent_id
        self.agent_name = self.config.agent_name
        self.version = self.config.version

        logger.info(
            f"Initialized {self.agent_name} agent v{self.version} (ID: {self.agent_id})"
        )

    def run(self, input_data: CondenserInput) -> CondenserOutput:
        """
        Execute condenser optimization analysis.

        This is the main entry point for the agent. It performs:
        1. Heat transfer analysis (HEI method)
        2. Cleanliness factor calculation and trending
        3. Vacuum system analysis
        4. Fouling analysis and prediction
        5. Air in-leakage assessment (if data provided)
        6. Optimization recommendations generation
        7. Cleaning schedule optimization
        8. Explainability report generation

        Args:
            input_data: Validated condenser input data

        Returns:
            CondenserOutput with complete analysis and recommendations

        Raises:
            ValueError: If input validation fails
            RuntimeError: If calculation fails
        """
        start_time = datetime.now()
        validation_errors: List[str] = []

        logger.info(f"Starting analysis for condenser {input_data.condenser_id}")

        try:
            # Step 1: Calculate heat transfer metrics
            heat_transfer = self._analyze_heat_transfer(input_data)
            logger.debug(f"Heat duty: {heat_transfer.heat_duty_kw:.0f} kW")

            # Step 2: Calculate cleanliness factor
            cleanliness = self._analyze_cleanliness(
                input_data, heat_transfer.calculated_u_actual
            )
            logger.debug(f"Cleanliness factor: {cleanliness.cleanliness_factor:.3f}")

            # Step 3: Analyze vacuum system
            vacuum = self._analyze_vacuum(input_data, heat_transfer)
            logger.debug(f"Vacuum efficiency: {vacuum.vacuum_efficiency_pct:.1f}%")

            # Step 4: Analyze fouling
            fouling = self._analyze_fouling(input_data, heat_transfer)
            logger.debug(f"Fouling severity: {fouling.severity}")

            # Step 5: Analyze air in-leakage (if data provided)
            air_leakage = None
            if input_data.air_leakage and input_data.air_leakage.measured_leakage_kg_hr:
                air_leakage = self._analyze_air_leakage(input_data)
                logger.debug(f"Air leakage severity: {air_leakage.severity}")

            # Step 6: Calculate overall efficiency score
            efficiency_score = self._calculate_efficiency_score(
                cleanliness, vacuum, fouling, air_leakage
            )
            performance_status = self._determine_performance_status(efficiency_score)
            logger.debug(f"Efficiency score: {efficiency_score:.1f}/100")

            # Step 7: Generate recommendations
            recommendations = self._generate_recommendations(
                cleanliness, vacuum, fouling, air_leakage
            )

            # Step 8: Generate cleaning schedule
            cleaning_schedule = self._generate_cleaning_schedule(
                input_data, cleanliness, fouling
            )

            # Step 9: Generate explainability report
            explainability = self._generate_explainability_report(
                heat_transfer, cleanliness, vacuum, fouling
            )

            # Step 10: Calculate provenance hash
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            provenance_hash = self._calculate_provenance_hash(
                input_data, heat_transfer, cleanliness, vacuum, fouling
            )

            # Validate output
            validation_status = "PASS"
            if cleanliness.cleanliness_factor < 0 or cleanliness.cleanliness_factor > 1.1:
                validation_errors.append("Cleanliness factor out of expected range")
                validation_status = "FAIL"

            # Build output
            output = CondenserOutput(
                condenser_id=input_data.condenser_id,
                assessment_timestamp=datetime.now(),
                heat_transfer=heat_transfer,
                cleanliness=cleanliness,
                vacuum=vacuum,
                fouling=fouling,
                air_leakage=air_leakage,
                overall_efficiency_score=efficiency_score,
                performance_status=performance_status,
                recommendations=recommendations,
                cleaning_schedule=cleaning_schedule,
                explainability=explainability,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
                validation_status=validation_status,
                validation_errors=validation_errors,
                agent_version=self.version,
            )

            logger.info(
                f"Completed analysis for {input_data.condenser_id} in {processing_time_ms:.1f}ms"
            )

            return output

        except Exception as e:
            logger.error(
                f"Analysis failed for {input_data.condenser_id}: {str(e)}",
                exc_info=True
            )
            raise RuntimeError(f"Condenser analysis failed: {str(e)}") from e

    def _analyze_heat_transfer(
        self,
        input_data: CondenserInput
    ) -> HeatTransferAnalysis:
        """
        Perform heat transfer analysis using HEI method.

        All calculations follow HEI Standards for Steam Surface Condensers.
        """
        design = input_data.design
        cw = input_data.cooling_water
        vac = input_data.vacuum

        # Get saturation temperature
        if vac.saturation_temp_c:
            t_sat = vac.saturation_temp_c
        else:
            t_sat = calculate_saturation_temperature(vac.absolute_pressure_kpa)

        # Calculate temperature metrics
        ttd = calculate_ttd(t_sat, cw.outlet_temp_c)
        cw_rise = calculate_cooling_water_rise(cw.inlet_temp_c, cw.outlet_temp_c)
        lmtd = calculate_lmtd(t_sat, cw.inlet_temp_c, cw.outlet_temp_c)

        # Calculate heat duty from water side
        # Q = m * Cp * dT
        water_density = 1000  # kg/m3 (approximate)
        cp_water = 4.18  # kJ/kg-K
        flow_kg_s = (cw.flow_rate_m3_hr / 3600) * water_density
        heat_duty = flow_kg_s * cp_water * cw_rise  # kW

        # Calculate actual U from Q = U * A * LMTD
        # U = Q / (A * LMTD)
        q_watts = heat_duty * 1000
        u_actual = q_watts / (design.surface_area_m2 * lmtd)

        # Get clean U from design or calculate
        u_clean = design.design_u_clean

        return HeatTransferAnalysis(
            calculated_u_actual=u_actual,
            u_clean=u_clean,
            lmtd=lmtd,
            heat_duty_kw=heat_duty,
            ttd_c=ttd,
            cooling_water_rise_c=cw_rise,
        )

    def _analyze_cleanliness(
        self,
        input_data: CondenserInput,
        u_actual: float
    ) -> CleanlinessAnalysis:
        """
        Analyze cleanliness factor with trending.
        """
        design = input_data.design

        # Calculate cleanliness factor
        cf = calculate_cleanliness_factor(u_actual, design.design_u_clean)

        # Get status and guidance
        status, guidance = assess_cleanliness_status(cf)

        # Calculate CF trend if fouling history available
        degradation_rate = None
        projected_cf = None
        days_to_threshold = None

        if input_data.fouling and input_data.fouling.fouling_history:
            try:
                # Convert fouling history to CF history approximation
                # This is simplified - production would track CF directly
                history = input_data.fouling.fouling_history
                if len(history) >= 2:
                    params = estimate_fouling_parameters_from_history(history)
                    if params['fouling_rate'] > 0:
                        # Approximate CF degradation from fouling rate
                        # CF drops roughly proportionally to fouling increase
                        degradation_rate = params['fouling_rate'] * design.design_u_clean / 1000
                        projected_cf = max(0.5, cf - degradation_rate * 30)
                        if degradation_rate > 0:
                            days_to_threshold = (cf - 0.75) / degradation_rate
            except Exception as e:
                logger.warning(f"Could not calculate CF trend: {e}")

        # Calculate CF score
        design_cf = get_design_cleanliness_factor(
            input_data.design.cooling_water_source.value
        )
        deviation = design_cf - cf
        cf_score = calculate_cf_score(cf, degradation_rate or 0, deviation)

        return CleanlinessAnalysis(
            cleanliness_factor=cf,
            status=status,
            guidance=guidance,
            degradation_rate_per_day=degradation_rate,
            projected_cf_30d=projected_cf,
            days_to_threshold=days_to_threshold,
            cf_score=cf_score,
        )

    def _analyze_vacuum(
        self,
        input_data: CondenserInput,
        heat_transfer: HeatTransferAnalysis
    ) -> VacuumAnalysis:
        """
        Analyze vacuum system performance.
        """
        vac = input_data.vacuum
        cw = input_data.cooling_water

        # Calculate theoretical vacuum
        theoretical_pressure = calculate_theoretical_vacuum(
            cw.inlet_temp_c,
            heat_transfer.cooling_water_rise_c,
            input_data.design.design_ttd_c
        )

        # Calculate deviation
        deviation, dev_status = calculate_vacuum_deviation(
            vac.absolute_pressure_kpa,
            theoretical_pressure
        )

        # Calculate efficiency
        efficiency = calculate_vacuum_efficiency(
            vac.absolute_pressure_kpa,
            theoretical_pressure
        )

        # Calculate vacuum in inches Hg
        vacuum_in_hg = calculate_vacuum_inches_hg(
            vac.absolute_pressure_kpa,
            vac.barometric_pressure_kpa
        )

        # Calculate power loss if turbine data available
        power_loss = None
        if input_data.turbine_power_mw:
            power_loss = calculate_power_loss_from_vacuum_degradation(
                vac.absolute_pressure_kpa,
                input_data.design.design_pressure_kpa,
                input_data.turbine_power_mw
            )

        return VacuumAnalysis(
            theoretical_pressure_kpa=theoretical_pressure,
            actual_pressure_kpa=vac.absolute_pressure_kpa,
            pressure_deviation_kpa=deviation,
            vacuum_efficiency_pct=efficiency,
            vacuum_in_hg=vacuum_in_hg,
            status=dev_status,
            power_loss_mw=power_loss,
        )

    def _analyze_fouling(
        self,
        input_data: CondenserInput,
        heat_transfer: HeatTransferAnalysis
    ) -> FoulingAnalysis:
        """
        Analyze tube fouling.
        """
        design = input_data.design

        # Calculate fouling resistance
        r_fouling = calculate_fouling_resistance(
            design.design_u_clean,
            heat_transfer.calculated_u_actual
        )

        # Get design fouling resistance
        r_design = get_standard_fouling_resistance(
            design.cooling_water_source.value
        )

        # Assess severity
        severity, ratio = assess_fouling_severity(r_fouling, r_design)

        # Calculate fouling rate if history available
        fouling_rate = None
        hours_to_threshold = None
        dominant_mechanism = None

        if input_data.fouling:
            if input_data.fouling.hours_since_cleaning > 0:
                fouling_rate = calculate_fouling_rate(
                    design.design_u_clean,
                    heat_transfer.calculated_u_actual,
                    input_data.fouling.hours_since_cleaning
                )

                # Calculate hours to cleaning threshold
                threshold = r_design * self.config.fouling_threshold_multiplier
                hours_to_threshold = calculate_hours_to_cleaning_threshold(
                    r_fouling,
                    fouling_rate,
                    threshold
                )

            if input_data.fouling.dominant_fouling_mechanism:
                dominant_mechanism = input_data.fouling.dominant_fouling_mechanism.value

        return FoulingAnalysis(
            fouling_resistance_m2k_w=r_fouling,
            design_fouling_m2k_w=r_design,
            fouling_ratio=ratio,
            severity=severity,
            fouling_rate_per_1000h=fouling_rate,
            hours_to_threshold=hours_to_threshold,
            dominant_mechanism=dominant_mechanism,
        )

    def _analyze_air_leakage(
        self,
        input_data: CondenserInput
    ) -> AirLeakageAnalysis:
        """
        Analyze air in-leakage.
        """
        air_data = input_data.air_leakage
        design = input_data.design

        # Assess severity
        heat_duty_kw = design.design_heat_duty_kw
        severity, normalized = assess_air_leakage_severity(
            air_data.measured_leakage_kg_hr,
            heat_duty_kw
        )

        # HEI limit (kg/hr)
        hei_limit = 0.0014 * heat_duty_kw  # Old condenser standard

        return AirLeakageAnalysis(
            measured_leakage_kg_hr=air_data.measured_leakage_kg_hr,
            normalized_leakage=normalized,
            severity=severity,
            hei_limit_kg_hr=hei_limit,
        )

    def _calculate_efficiency_score(
        self,
        cleanliness: CleanlinessAnalysis,
        vacuum: VacuumAnalysis,
        fouling: FoulingAnalysis,
        air_leakage: Optional[AirLeakageAnalysis]
    ) -> float:
        """
        Calculate overall condenser efficiency score (0-100).
        """
        # Component weights
        cf_weight = 0.35
        vacuum_weight = 0.30
        fouling_weight = 0.20
        air_weight = 0.15

        # CF score (already 0-100)
        cf_score = cleanliness.cf_score

        # Vacuum score (efficiency-based)
        vacuum_score = min(100, vacuum.vacuum_efficiency_pct)

        # Fouling score (inverse of ratio)
        fouling_score = max(0, 100 - (fouling.fouling_ratio - 1) * 100)

        # Air leakage score
        if air_leakage:
            if air_leakage.severity == "EXCELLENT":
                air_score = 100
            elif air_leakage.severity == "ACCEPTABLE":
                air_score = 80
            elif air_leakage.severity == "HIGH":
                air_score = 50
            else:
                air_score = 20
        else:
            # Assume acceptable if no data
            air_score = 80

        # Weighted total
        total = (
            cf_weight * cf_score +
            vacuum_weight * vacuum_score +
            fouling_weight * fouling_score +
            air_weight * air_score
        )

        return max(0, min(100, total))

    def _determine_performance_status(
        self,
        efficiency_score: float
    ) -> str:
        """
        Determine overall performance status from efficiency score.
        """
        if efficiency_score >= 90:
            return "OPTIMAL"
        elif efficiency_score >= 80:
            return "GOOD"
        elif efficiency_score >= 70:
            return "DEGRADED"
        elif efficiency_score >= 60:
            return "POOR"
        else:
            return "CRITICAL"

    def _generate_recommendations(
        self,
        cleanliness: CleanlinessAnalysis,
        vacuum: VacuumAnalysis,
        fouling: FoulingAnalysis,
        air_leakage: Optional[AirLeakageAnalysis]
    ) -> List[OptimizationRecommendation]:
        """
        Generate prioritized optimization recommendations.
        """
        recommendations = []

        # Cleanliness recommendations
        cf_recs = generate_cf_recommendations(
            cleanliness.cleanliness_factor,
            cleanliness.degradation_rate_per_day or 0,
            cleanliness.days_to_threshold or float('inf')
        )
        for rec in cf_recs:
            recommendations.append(OptimizationRecommendation(
                action=rec["action"],
                priority=OptimizationPriority(rec["priority"]),
                reason=rec["reason"],
                category="cleanliness",
            ))

        # Vacuum recommendations
        vac_recs = generate_vacuum_recommendations(
            vacuum.actual_pressure_kpa,
            vacuum.theoretical_pressure_kpa,
            air_leakage.severity if air_leakage else "ACCEPTABLE",
            cleanliness.cleanliness_factor,
            0  # TTD handled separately
        )
        for rec in vac_recs:
            recommendations.append(OptimizationRecommendation(
                action=rec["action"],
                priority=OptimizationPriority(rec["priority"]),
                reason=rec["reason"],
                category="vacuum",
            ))

        # Fouling recommendations
        fouling_recs = generate_fouling_recommendations(
            fouling.severity,
            fouling.dominant_mechanism or "unknown",
            fouling.fouling_rate_per_1000h or 0,
            fouling.hours_to_threshold or float('inf')
        )
        for rec in fouling_recs:
            recommendations.append(OptimizationRecommendation(
                action=rec["action"],
                priority=OptimizationPriority(rec["priority"]),
                reason=rec["reason"],
                category="fouling",
            ))

        # Sort by priority
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
        recommendations.sort(key=lambda r: priority_order.get(r.priority.value, 5))

        return recommendations

    def _generate_cleaning_schedule(
        self,
        input_data: CondenserInput,
        cleanliness: CleanlinessAnalysis,
        fouling: FoulingAnalysis
    ) -> Optional[CleaningSchedule]:
        """
        Generate optimal cleaning schedule.
        """
        # Only generate if cleaning is needed
        if cleanliness.status in ["EXCELLENT", "GOOD"]:
            return None

        # Determine urgency
        if cleanliness.status == "POOR" or fouling.severity == "SEVERE":
            urgency = "URGENT"
            days_ahead = 14
        elif cleanliness.status == "MARGINAL" or fouling.severity == "HEAVY":
            urgency = "SOON"
            days_ahead = 30
        else:
            urgency = "PLANNED"
            days_ahead = 60

        # Use fouling prediction if available
        if cleanliness.days_to_threshold and cleanliness.days_to_threshold < days_ahead:
            days_ahead = max(7, int(cleanliness.days_to_threshold * 0.8))

        recommended_date = date.today() + __import__('datetime').timedelta(days=days_ahead)

        # Select cleaning method based on dominant mechanism
        if fouling.dominant_mechanism == "biological":
            method = CleaningMethod.CHEMICAL_BIODISPERSANT
        elif fouling.dominant_mechanism == "scaling":
            method = CleaningMethod.CHEMICAL_ACID
        else:
            method = CleaningMethod.MECHANICAL_BALL

        # Expected CF after cleaning
        effectiveness = CLEANING_EFFECTIVENESS.get(method.value, 0.85)
        cf_recovery = (1.0 - cleanliness.cleanliness_factor) * effectiveness
        expected_cf = cleanliness.cleanliness_factor + cf_recovery

        # Power recovery estimate
        power_recovery = None
        if input_data.turbine_power_mw:
            benefit = calculate_cleaning_benefit_cf(
                cleanliness.cleanliness_factor,
                expected_cf,
                input_data.turbine_power_mw
            )
            power_recovery = benefit["power_recovery_mw"]

        return CleaningSchedule(
            recommended_date=recommended_date,
            urgency=urgency,
            method=method,
            expected_cf_after=expected_cf,
            estimated_power_recovery_mw=power_recovery,
        )

    def _generate_explainability_report(
        self,
        heat_transfer: HeatTransferAnalysis,
        cleanliness: CleanlinessAnalysis,
        vacuum: VacuumAnalysis,
        fouling: FoulingAnalysis
    ) -> ExplainabilityReport:
        """
        Generate SHAP/LIME-style explainability report.

        This provides human-interpretable explanations for the
        optimization decisions without using actual ML models.
        """
        # Feature importance (normalized)
        importance = {
            "cleanliness_factor": 0.35,
            "vacuum_efficiency": 0.25,
            "fouling_severity": 0.20,
            "ttd_deviation": 0.12,
            "air_leakage": 0.08,
        }

        # Identify key drivers
        key_drivers = []

        if cleanliness.status in ["MARGINAL", "POOR", "CRITICAL"]:
            key_drivers.append(
                f"Low cleanliness factor ({cleanliness.cleanliness_factor:.2f}) "
                f"is the primary performance limiter"
            )

        if vacuum.status in ["MARGINAL", "POOR"]:
            key_drivers.append(
                f"Vacuum deviation of {vacuum.pressure_deviation_kpa:.2f} kPa "
                f"reduces turbine efficiency"
            )

        if fouling.severity in ["HEAVY", "SEVERE"]:
            key_drivers.append(
                f"Fouling at {fouling.fouling_ratio:.1f}x design level "
                f"significantly impacts heat transfer"
            )

        if heat_transfer.ttd_c > 5.0:
            key_drivers.append(
                f"Elevated TTD of {heat_transfer.ttd_c:.1f}C indicates "
                f"condenser performance degradation"
            )

        if not key_drivers:
            key_drivers.append(
                "Condenser is operating within acceptable parameters"
            )

        # Generate rationale
        rationale = self._build_optimization_rationale(
            cleanliness, vacuum, fouling
        )

        # Calculate confidence based on data quality
        confidence = 0.85  # Base confidence for HEI method
        if cleanliness.degradation_rate_per_day is not None:
            confidence += 0.05  # Higher confidence with trend data
        if fouling.fouling_rate_per_1000h is not None:
            confidence += 0.05

        return ExplainabilityReport(
            feature_importance=importance,
            key_drivers=key_drivers,
            optimization_rationale=rationale,
            confidence_score=min(0.95, confidence),
        )

    def _build_optimization_rationale(
        self,
        cleanliness: CleanlinessAnalysis,
        vacuum: VacuumAnalysis,
        fouling: FoulingAnalysis
    ) -> str:
        """
        Build human-readable optimization rationale.
        """
        parts = []

        parts.append(
            f"Analysis performed using HEI Standards methodology. "
            f"Current cleanliness factor is {cleanliness.cleanliness_factor:.3f} "
            f"({cleanliness.status}). "
        )

        if cleanliness.status not in ["EXCELLENT", "GOOD"]:
            parts.append(
                f"Tube fouling has reduced heat transfer capability by "
                f"{(1 - cleanliness.cleanliness_factor) * 100:.0f}%. "
            )

        parts.append(
            f"Vacuum system is operating at {vacuum.vacuum_efficiency_pct:.1f}% "
            f"of theoretical efficiency with {vacuum.pressure_deviation_kpa:.2f} kPa "
            f"deviation from ideal. "
        )

        if vacuum.power_loss_mw:
            parts.append(
                f"This corresponds to approximately {vacuum.power_loss_mw:.2f} MW "
                f"of lost generation capacity. "
            )

        if fouling.hours_to_threshold:
            if fouling.hours_to_threshold < 2000:
                parts.append(
                    f"At current fouling rate, cleaning threshold will be reached "
                    f"in approximately {fouling.hours_to_threshold:.0f} operating hours. "
                    f"Cleaning should be scheduled soon. "
                )
            else:
                parts.append(
                    f"Fouling rate is acceptable with {fouling.hours_to_threshold:.0f} "
                    f"hours until cleaning threshold. "
                )

        return "".join(parts)

    def _calculate_provenance_hash(
        self,
        input_data: CondenserInput,
        heat_transfer: HeatTransferAnalysis,
        cleanliness: CleanlinessAnalysis,
        vacuum: VacuumAnalysis,
        fouling: FoulingAnalysis
    ) -> str:
        """
        Calculate SHA-256 hash for complete audit trail.

        This hash provides cryptographic proof of the input data
        and calculated results for regulatory compliance.
        """
        provenance_data = {
            "input": input_data.json(),
            "heat_transfer": heat_transfer.json(),
            "cleanliness": cleanliness.json(),
            "vacuum": vacuum.json(),
            "fouling": fouling.json(),
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "calculation_method": "HEI_STANDARDS",
        }

        provenance_str = str(provenance_data)
        return hashlib.sha256(provenance_str.encode('utf-8')).hexdigest()
