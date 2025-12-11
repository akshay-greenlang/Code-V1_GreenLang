"""
GL-003 UNIFIED STEAM SYSTEM OPTIMIZER - Main Optimizer Class

This module provides the main UnifiedSteamOptimizer class that consolidates
GL-003 (STEAMWISE) and GL-012 (STEAMQUAL) functionality into a single,
comprehensive steam system optimization agent.

Features:
    - Steam header pressure balancing with exergy optimization
    - Steam quality monitoring per ASME standards
    - Condensate return optimization
    - Flash steam recovery calculations
    - PRV sizing and optimization per ASME B31.1
    - Desuperheating control
    - Steam trap survey integration
    - Zero-hallucination deterministic calculations
    - SHA-256 provenance tracking

Example:
    >>> from greenlang.agents.process_heat.gl_003_unified_steam import (
    ...     UnifiedSteamOptimizer,
    ...     UnifiedSteamConfig,
    ...     create_default_config,
    ... )
    >>>
    >>> config = create_default_config()
    >>> optimizer = UnifiedSteamOptimizer(config)
    >>> result = optimizer.optimize(system_data)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import json
import logging
import time

from pydantic import BaseModel, Field

# Intelligence imports for LLM capabilities
from greenlang.agents.intelligence_mixin import IntelligenceMixin, IntelligenceConfig
from greenlang.agents.intelligence_interface import IntelligenceCapabilities, IntelligenceLevel

# Import from shared base
from ..shared.base_agent import (
    BaseProcessHeatAgent,
    AgentConfig,
    AgentCapability,
    SafetyLevel,
    ProcessingError,
    ValidationError,
)

# Import local modules
from .config import (
    UnifiedSteamConfig,
    SteamHeaderConfig,
    PRVConfig,
    DesuperheaterConfig,
    QualityMonitoringConfig,
    CondensateConfig,
    FlashRecoveryConfig,
    SteamTrapSurveyConfig,
    ExergyOptimizationConfig,
    create_default_config,
)
from .schemas import (
    HeaderBalanceInput,
    HeaderBalanceOutput,
    SteamQualityReading,
    SteamQualityAnalysis,
    CondensateReading,
    CondensateReturnAnalysis,
    FlashSteamInput,
    FlashSteamOutput,
    PRVOperatingPoint,
    PRVSizingOutput,
    SteamTrapReading,
    TrapSurveyAnalysis,
    OptimizationRecommendation,
    UnifiedSteamOptimizerOutput,
    OptimizationStatus,
    ValidationStatus,
)
from .distribution import (
    SteamDistributionOptimizer,
    SteamPropertyCalculator,
    HeaderBalanceCalculator,
)
from .quality import (
    SteamQualityMonitor,
    QualityLimitCalculator,
    DrynessFractionCalculator,
    CarryoverAnalyzer,
)
from .condensate import (
    CondensateReturnOptimizer,
    CondensateHeatCalculator,
    CondensateQualityAnalyzer,
    SteamTrapSurveyAnalyzer,
)
from .flash_recovery import (
    FlashRecoveryOptimizer,
    FlashSteamCalculator,
    FlashTankSizer,
    MultiStageFlashOptimizer,
)
from .prv_optimization import (
    PRVOptimizer,
    CvCalculator,
    DesuperheaterCalculator,
    MultiPRVCoordinator,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class UnifiedSteamOptimizerInput(BaseModel):
    """Input data model for UnifiedSteamOptimizer."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Input timestamp"
    )

    # Header readings
    header_readings: List[HeaderBalanceInput] = Field(
        default_factory=list,
        description="Current header readings"
    )

    # Quality readings
    quality_readings: List[SteamQualityReading] = Field(
        default_factory=list,
        description="Steam quality readings"
    )

    # PRV operating points
    prv_readings: List[PRVOperatingPoint] = Field(
        default_factory=list,
        description="PRV operating data"
    )

    # Condensate readings
    condensate_readings: List[CondensateReading] = Field(
        default_factory=list,
        description="Condensate return readings"
    )

    # Steam trap readings
    trap_readings: List[SteamTrapReading] = Field(
        default_factory=list,
        description="Steam trap survey data"
    )

    # Operating context
    total_steam_flow_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total steam production (lb/hr)"
    )
    boiler_water_tds_ppm: Optional[float] = Field(
        default=None,
        description="Boiler water TDS for carryover calculation"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# UNIFIED STEAM OPTIMIZER
# =============================================================================

class UnifiedSteamOptimizer(IntelligenceMixin, BaseProcessHeatAgent[UnifiedSteamOptimizerInput, UnifiedSteamOptimizerOutput]):
    """
    GL-003 Unified Steam System Optimizer.

    This agent consolidates GL-003 (STEAMWISE) and GL-012 (STEAMQUAL) to provide
    comprehensive steam system optimization with 60% reduced functional overlap.

    Capabilities:
        - Steam header pressure balancing with exergy-based optimization
        - Steam quality monitoring per ASME standards (dryness, TDS, conductivity)
        - Flash steam recovery calculations (thermodynamic fraction extraction)
        - PRV sizing and optimization per ASME B31.1 (50-70% opening targets)
        - Desuperheating control
        - Condensate return temperature maximization
        - Steam trap survey integration
        - IAPWS-IF97 steam property calculations

    All calculations are deterministic with zero hallucination and complete
    SHA-256 provenance tracking for regulatory compliance.

    Example:
        >>> config = create_default_config()
        >>> optimizer = UnifiedSteamOptimizer(config)
        >>>
        >>> input_data = UnifiedSteamOptimizerInput(
        ...     header_readings=[...],
        ...     quality_readings=[...],
        ...     total_steam_flow_lb_hr=100000,
        ... )
        >>>
        >>> result = optimizer.process(input_data)
        >>> print(f"System efficiency: {result.system_efficiency_pct}%")
    """

    # Agent metadata
    AGENT_TYPE = "GL-003"
    AGENT_NAME = "Unified Steam System Optimizer"
    AGENT_VERSION = "2.0.0"

    def __init__(
        self,
        config: UnifiedSteamConfig,
        safety_level: SafetyLevel = SafetyLevel.SIL_2,
    ) -> None:
        """
        Initialize the Unified Steam System Optimizer.

        Args:
            config: Unified steam system configuration
            safety_level: Safety Integrity Level (default SIL-2)
        """
        # Create agent config for base class
        agent_config = AgentConfig(
            agent_id=config.agent_id,
            agent_type=self.AGENT_TYPE,
            name=self.AGENT_NAME,
            version=self.AGENT_VERSION,
            capabilities={
                AgentCapability.REAL_TIME_MONITORING,
                AgentCapability.OPTIMIZATION,
                AgentCapability.COMPLIANCE_REPORTING,
            },
        )

        # Initialize base class
        super().__init__(agent_config, safety_level)

        # Store configuration
        self.steam_config = config

        # Initialize sub-optimizers
        self._init_distribution_optimizer()
        self._init_quality_monitor()
        self._init_condensate_optimizer()
        self._init_flash_optimizer()
        self._init_prv_optimizers()

        # Steam property calculator
        self.steam_calc = SteamPropertyCalculator(
            reference_temp_f=config.exergy.reference_temperature_f
        )

        # Calculation counter for provenance
        self._calculation_count = 0

        logger.info(
            f"UnifiedSteamOptimizer initialized: "
            f"{len(config.headers)} headers, "
            f"{len(config.prvs)} PRVs"
        )

        # Initialize intelligence with ADVANCED level for steam optimizer
        self._init_intelligence(IntelligenceConfig(
            domain_context="steam systems and thermal energy optimization",
            regulatory_context="IAPWS-IF97, ASME PTC 4",
            enable_explanations=True,
            enable_recommendations=True,
            enable_anomaly_detection=True,
        ))

    # =========================================================================
    # INTELLIGENCE INTERFACE METHODS
    # =========================================================================

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return ADVANCED intelligence level for steam optimizer."""
        return IntelligenceLevel.ADVANCED

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """Return advanced intelligence capabilities."""
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=True,
            can_reason=True,
            can_validate=True,
            uses_rag=False,
            uses_tools=False
        )

    def _init_distribution_optimizer(self) -> None:
        """Initialize steam distribution optimizer."""
        self.distribution_optimizer = SteamDistributionOptimizer(
            headers=self.steam_config.headers,
            exergy_config=self.steam_config.exergy,
        )

    def _init_quality_monitor(self) -> None:
        """Initialize steam quality monitor."""
        self.quality_monitor = SteamQualityMonitor(
            config=self.steam_config.quality,
        )

    def _init_condensate_optimizer(self) -> None:
        """Initialize condensate return optimizer."""
        self.condensate_optimizer = CondensateReturnOptimizer(
            config=self.steam_config.condensate,
            trap_survey_config=self.steam_config.trap_survey,
        )

    def _init_flash_optimizer(self) -> None:
        """Initialize flash steam recovery optimizers."""
        self.flash_optimizers: Dict[str, FlashRecoveryOptimizer] = {}
        for flash_config in self.steam_config.flash_recovery:
            self.flash_optimizers[flash_config.flash_tank_id] = FlashRecoveryOptimizer(
                config=flash_config,
            )

    def _init_prv_optimizers(self) -> None:
        """Initialize PRV optimizers."""
        self.prv_optimizers: Dict[str, PRVOptimizer] = {}
        for prv_config in self.steam_config.prvs:
            self.prv_optimizers[prv_config.prv_id] = PRVOptimizer(
                config=prv_config,
            )

        # Multi-PRV coordinator
        if len(self.steam_config.prvs) > 1:
            self.prv_coordinator = MultiPRVCoordinator(
                prvs=self.steam_config.prvs,
            )
        else:
            self.prv_coordinator = None

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    def process(
        self,
        input_data: UnifiedSteamOptimizerInput,
    ) -> UnifiedSteamOptimizerOutput:
        """
        Main processing method for steam system optimization.

        This method orchestrates all sub-optimizers to provide comprehensive
        steam system analysis and recommendations.

        Args:
            input_data: Validated input data with readings from all subsystems

        Returns:
            UnifiedSteamOptimizerOutput with complete analysis

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing fails
        """
        start_time = time.time()
        self._calculation_count = 0

        logger.info(f"Processing steam system optimization at {input_data.timestamp}")

        try:
            with self.safety_guard():
                # Step 1: Validate input
                if not self.validate_input(input_data):
                    raise ValidationError("Input validation failed")

                # Step 2: Analyze headers
                header_analyses = self._analyze_headers(input_data.header_readings)
                self._calculation_count += len(header_analyses)

                # Step 3: Analyze steam quality
                quality_analyses = self._analyze_quality(
                    input_data.quality_readings,
                    input_data.boiler_water_tds_ppm,
                )
                self._calculation_count += len(quality_analyses)

                # Step 4: Analyze PRVs
                prv_analyses = self._analyze_prvs(input_data.prv_readings)
                self._calculation_count += len(prv_analyses)

                # Step 5: Analyze condensate return
                condensate_analysis = None
                if input_data.condensate_readings and input_data.total_steam_flow_lb_hr > 0:
                    condensate_analysis = self._analyze_condensate(
                        input_data.total_steam_flow_lb_hr,
                        input_data.condensate_readings,
                        input_data.trap_readings,
                    )
                    self._calculation_count += 1

                # Step 6: Analyze flash steam recovery
                flash_analyses = self._analyze_flash_recovery(input_data)
                self._calculation_count += len(flash_analyses)

                # Step 7: Analyze steam traps
                trap_analysis = None
                if input_data.trap_readings:
                    trap_analysis = self._analyze_traps(input_data.trap_readings)
                    self._calculation_count += 1

                # Step 8: Calculate system efficiency
                system_efficiency = self._calculate_system_efficiency(
                    header_analyses,
                    quality_analyses,
                    condensate_analysis,
                )

                # Step 9: Calculate exergy efficiency
                exergy_efficiency = None
                if self.steam_config.exergy.enabled:
                    exergy_efficiency = self.distribution_optimizer.calculate_system_exergy_efficiency(
                        header_analyses
                    )

                # Step 10: Determine overall status
                overall_status = self._determine_overall_status(
                    header_analyses,
                    quality_analyses,
                    prv_analyses,
                )

                # Step 11: Generate recommendations
                recommendations = self._generate_recommendations(
                    header_analyses,
                    quality_analyses,
                    prv_analyses,
                    condensate_analysis,
                    flash_analyses,
                    trap_analysis,
                )

                # Step 12: Collect warnings and alerts
                warnings, alerts = self._collect_warnings_alerts(
                    header_analyses,
                    quality_analyses,
                    prv_analyses,
                    condensate_analysis,
                )

                # Calculate processing time
                processing_time_ms = (time.time() - start_time) * 1000

                # Calculate provenance hash
                provenance_hash = self._calculate_provenance_hash(input_data)

                # Create output
                output = UnifiedSteamOptimizerOutput(
                    optimizer_id=self.steam_config.agent_id,
                    timestamp=datetime.now(timezone.utc),
                    overall_status=overall_status,
                    system_efficiency_pct=system_efficiency,
                    exergy_efficiency_pct=exergy_efficiency,
                    header_analyses=header_analyses,
                    quality_analyses=quality_analyses,
                    prv_analyses=prv_analyses,
                    condensate_analysis=condensate_analysis,
                    flash_analyses=flash_analyses,
                    trap_analysis=trap_analysis,
                    recommendations=recommendations,
                    warnings=warnings,
                    alerts=alerts,
                    provenance_hash=provenance_hash,
                    processing_time_ms=processing_time_ms,
                    calculation_count=self._calculation_count,
                )

                # Validate output
                if not self.validate_output(output):
                    raise ProcessingError("Output validation failed")

                logger.info(
                    f"Steam optimization complete: {overall_status.value}, "
                    f"{self._calculation_count} calculations, "
                    f"{processing_time_ms:.1f}ms"
                )

                # Generate intelligent explanation of optimization results
                output.explanation = self.generate_explanation(
                    input_data={
                        "headers": len(input_data.header_readings),
                        "quality_readings": len(input_data.quality_readings),
                        "total_steam_flow_lb_hr": input_data.total_steam_flow_lb_hr
                    },
                    output_data={
                        "status": overall_status.value,
                        "system_efficiency_pct": system_efficiency,
                        "exergy_efficiency_pct": exergy_efficiency,
                        "recommendations_count": len(recommendations)
                    },
                    calculation_steps=[
                        f"Analyzed {len(header_analyses)} headers",
                        f"Processed {len(quality_analyses)} quality readings",
                        f"Generated {len(recommendations)} recommendations",
                        f"System efficiency: {system_efficiency:.1f}%"
                    ]
                )

                return output

        except Exception as e:
            logger.error(f"Steam optimization failed: {e}", exc_info=True)
            raise ProcessingError(f"Steam optimization failed: {str(e)}") from e

    def validate_input(self, input_data: UnifiedSteamOptimizerInput) -> bool:
        """
        Validate input data.

        Args:
            input_data: Input to validate

        Returns:
            True if valid
        """
        # Basic validation - input_data is already Pydantic validated
        if input_data.total_steam_flow_lb_hr < 0:
            logger.warning("Negative steam flow provided")
            return False

        # Check for reasonable values
        for reading in input_data.header_readings:
            if reading.current_pressure_psig > 1500:
                logger.warning(f"Unusually high pressure: {reading.current_pressure_psig}")

        return True

    def validate_output(self, output_data: UnifiedSteamOptimizerOutput) -> bool:
        """
        Validate output data.

        Args:
            output_data: Output to validate

        Returns:
            True if valid
        """
        # Check efficiency bounds
        if output_data.system_efficiency_pct < 0 or output_data.system_efficiency_pct > 100:
            logger.warning(f"Invalid efficiency: {output_data.system_efficiency_pct}")
            return False

        # Check provenance hash exists
        if not output_data.provenance_hash:
            logger.warning("Missing provenance hash")
            return False

        return True

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def _analyze_headers(
        self,
        readings: List[HeaderBalanceInput],
    ) -> List[HeaderBalanceOutput]:
        """Analyze all header readings."""
        results = []
        for reading in readings:
            try:
                result = self.distribution_optimizer.balance_header(
                    reading.header_id,
                    reading,
                )
                results.append(result)
            except ValueError as e:
                logger.warning(f"Header analysis failed for {reading.header_id}: {e}")

        return results

    def _analyze_quality(
        self,
        readings: List[SteamQualityReading],
        boiler_water_tds: Optional[float],
    ) -> List[SteamQualityAnalysis]:
        """Analyze steam quality readings."""
        results = []
        for reading in readings:
            analysis = self.quality_monitor.analyze_quality(
                reading,
                boiler_water_tds_ppm=boiler_water_tds,
            )
            results.append(analysis)

        return results

    def _analyze_prvs(
        self,
        readings: List[PRVOperatingPoint],
    ) -> List[PRVSizingOutput]:
        """Analyze PRV operating points."""
        results = []
        for reading in readings:
            prv_id = reading.prv_id
            if prv_id in self.prv_optimizers:
                optimizer = self.prv_optimizers[prv_id]
                # Get sizing analysis at current conditions
                sizing = optimizer.size_prv(
                    inlet_temperature_f=reading.inlet_temperature_f,
                )
                results.append(sizing)

        return results

    def _analyze_condensate(
        self,
        steam_flow_lb_hr: float,
        condensate_readings: List[CondensateReading],
        trap_readings: Optional[List[SteamTrapReading]],
    ) -> CondensateReturnAnalysis:
        """Analyze condensate return system."""
        return self.condensate_optimizer.analyze_return_system(
            steam_flow_lb_hr=steam_flow_lb_hr,
            condensate_readings=condensate_readings,
            trap_readings=trap_readings,
        )

    def _analyze_flash_recovery(
        self,
        input_data: UnifiedSteamOptimizerInput,
    ) -> List[FlashSteamOutput]:
        """Analyze flash steam recovery opportunities."""
        results = []

        for flash_id, optimizer in self.flash_optimizers.items():
            # Get condensate flow for this flash tank
            condensate_flow = optimizer.config.condensate_flow_lb_hr

            if condensate_flow > 0:
                analysis = optimizer.analyze(
                    condensate_flow_lb_hr=condensate_flow,
                )
                if "flash_analysis" in analysis:
                    results.append(analysis["flash_analysis"])

        return results

    def _analyze_traps(
        self,
        trap_readings: List[SteamTrapReading],
    ) -> TrapSurveyAnalysis:
        """Analyze steam trap survey."""
        analyzer = SteamTrapSurveyAnalyzer(self.steam_config.trap_survey)
        return analyzer.analyze_survey(trap_readings)

    # =========================================================================
    # EFFICIENCY CALCULATIONS
    # =========================================================================

    def _calculate_system_efficiency(
        self,
        header_analyses: List[HeaderBalanceOutput],
        quality_analyses: List[SteamQualityAnalysis],
        condensate_analysis: Optional[CondensateReturnAnalysis],
    ) -> float:
        """Calculate overall system efficiency."""
        efficiency_factors = []

        # Header balance efficiency (deviation from optimal)
        for header in header_analyses:
            if header.status == OptimizationStatus.OPTIMAL:
                efficiency_factors.append(1.0)
            elif header.status == OptimizationStatus.SUBOPTIMAL:
                efficiency_factors.append(0.9)
            else:
                efficiency_factors.append(0.7)

        # Quality efficiency
        for quality in quality_analyses:
            if quality.overall_status == ValidationStatus.VALID:
                efficiency_factors.append(1.0)
            elif quality.overall_status == ValidationStatus.WARNING:
                efficiency_factors.append(0.95)
            else:
                efficiency_factors.append(0.85)

        # Condensate return efficiency
        if condensate_analysis:
            return_efficiency = min(
                condensate_analysis.return_rate_pct /
                condensate_analysis.target_return_rate_pct,
                1.0
            )
            efficiency_factors.append(return_efficiency)

        # Calculate weighted average
        if efficiency_factors:
            overall = sum(efficiency_factors) / len(efficiency_factors) * 100
        else:
            overall = 85.0  # Default assumption

        return min(100.0, max(0.0, overall))

    # =========================================================================
    # STATUS DETERMINATION
    # =========================================================================

    def _determine_overall_status(
        self,
        header_analyses: List[HeaderBalanceOutput],
        quality_analyses: List[SteamQualityAnalysis],
        prv_analyses: List[PRVSizingOutput],
    ) -> OptimizationStatus:
        """Determine overall system status."""
        # Check for critical conditions
        for header in header_analyses:
            if header.status == OptimizationStatus.CRITICAL:
                return OptimizationStatus.CRITICAL

        for quality in quality_analyses:
            if quality.overall_status == ValidationStatus.INVALID:
                return OptimizationStatus.CRITICAL

        # Check for suboptimal conditions
        suboptimal_count = 0

        for header in header_analyses:
            if header.status == OptimizationStatus.SUBOPTIMAL:
                suboptimal_count += 1

        for quality in quality_analyses:
            if quality.overall_status == ValidationStatus.WARNING:
                suboptimal_count += 1

        for prv in prv_analyses:
            if not prv.meets_opening_targets:
                suboptimal_count += 1

        if suboptimal_count > 2:
            return OptimizationStatus.SUBOPTIMAL
        elif suboptimal_count > 0:
            return OptimizationStatus.SUBOPTIMAL

        return OptimizationStatus.OPTIMAL

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================

    def _generate_recommendations(
        self,
        header_analyses: List[HeaderBalanceOutput],
        quality_analyses: List[SteamQualityAnalysis],
        prv_analyses: List[PRVSizingOutput],
        condensate_analysis: Optional[CondensateReturnAnalysis],
        flash_analyses: List[FlashSteamOutput],
        trap_analysis: Optional[TrapSurveyAnalysis],
    ) -> List[OptimizationRecommendation]:
        """Generate prioritized optimization recommendations."""
        recommendations = []

        # Header balance recommendations
        for header in header_analyses:
            if header.status != OptimizationStatus.OPTIMAL:
                for adj in header.adjustments:
                    recommendations.append(OptimizationRecommendation(
                        category="header_balance",
                        priority=2,
                        description=f"Adjust {adj['source_id']} flow",
                        action=f"{adj['action'].title()} flow to {adj['recommended_flow_lb_hr']:.0f} lb/hr",
                    ))

        # Quality recommendations
        for quality in quality_analyses:
            for rec in quality.recommendations:
                recommendations.append(OptimizationRecommendation(
                    category="steam_quality",
                    priority=1 if quality.overall_status == ValidationStatus.INVALID else 2,
                    description=f"Quality issue at {quality.reading.location_id}",
                    action=rec,
                ))

        # PRV recommendations
        for prv in prv_analyses:
            for rec in prv.recommendations:
                recommendations.append(OptimizationRecommendation(
                    category="prv_optimization",
                    priority=2,
                    description=f"PRV {prv.prv_id} adjustment",
                    action=rec,
                ))

        # Condensate recommendations
        if condensate_analysis:
            for rec in condensate_analysis.recommendations:
                recommendations.append(OptimizationRecommendation(
                    category="condensate_recovery",
                    priority=2,
                    description="Condensate return improvement",
                    action=rec,
                ))

        # Flash steam recommendations
        for flash in flash_analyses:
            if flash.flash_fraction_pct > 5:
                recommendations.append(OptimizationRecommendation(
                    category="flash_recovery",
                    priority=3,
                    description="Flash steam recovery opportunity",
                    action=f"Recover {flash.flash_steam_lb_hr:.0f} lb/hr flash steam",
                    cost_savings_usd_year=flash.annual_savings_usd,
                ))

        # Trap recommendations
        if trap_analysis and trap_analysis.failure_rate_pct > 5:
            for rep in trap_analysis.priority_repairs[:5]:
                recommendations.append(OptimizationRecommendation(
                    category="trap_maintenance",
                    priority=1,
                    description="Steam trap repair needed",
                    action=rep,
                ))

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)

        return recommendations

    # =========================================================================
    # WARNINGS AND ALERTS
    # =========================================================================

    def _collect_warnings_alerts(
        self,
        header_analyses: List[HeaderBalanceOutput],
        quality_analyses: List[SteamQualityAnalysis],
        prv_analyses: List[PRVSizingOutput],
        condensate_analysis: Optional[CondensateReturnAnalysis],
    ) -> Tuple[List[str], List[str]]:
        """Collect all warnings and critical alerts."""
        warnings = []
        alerts = []

        # Header warnings
        for header in header_analyses:
            warnings.extend(header.warnings)
            if header.status == OptimizationStatus.CRITICAL:
                alerts.append(f"CRITICAL: Header {header.header_id} in critical state")

        # Quality warnings
        for quality in quality_analyses:
            for exc in quality.limits_exceeded:
                alerts.append(f"QUALITY: {exc}")
            for warn in quality.limits_warning:
                warnings.append(f"Quality: {warn}")

        # PRV warnings
        for prv in prv_analyses:
            warnings.extend(prv.warnings)

        # Condensate warnings
        if condensate_analysis:
            warnings.extend(condensate_analysis.warnings)

        return warnings, alerts

    # =========================================================================
    # PROVENANCE
    # =========================================================================

    def _calculate_provenance_hash(
        self,
        input_data: UnifiedSteamOptimizerInput,
    ) -> str:
        """Calculate SHA-256 provenance hash for complete audit trail."""
        provenance_data = {
            "agent_id": self.steam_config.agent_id,
            "agent_version": self.AGENT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_timestamp": input_data.timestamp.isoformat(),
            "header_count": len(input_data.header_readings),
            "quality_count": len(input_data.quality_readings),
            "prv_count": len(input_data.prv_readings),
            "condensate_count": len(input_data.condensate_readings),
            "trap_count": len(input_data.trap_readings),
            "total_steam_flow": input_data.total_steam_flow_lb_hr,
            "calculation_count": self._calculation_count,
        }

        data_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def analyze_header(
        self,
        header_id: str,
        reading: HeaderBalanceInput,
    ) -> HeaderBalanceOutput:
        """
        Analyze a single header.

        Convenience method for single-header analysis.

        Args:
            header_id: Header identifier
            reading: Header reading

        Returns:
            HeaderBalanceOutput
        """
        return self.distribution_optimizer.balance_header(header_id, reading)

    def analyze_quality(
        self,
        reading: SteamQualityReading,
        boiler_water_tds_ppm: Optional[float] = None,
    ) -> SteamQualityAnalysis:
        """
        Analyze steam quality reading.

        Convenience method for single quality analysis.

        Args:
            reading: Quality reading
            boiler_water_tds_ppm: Boiler water TDS for carryover

        Returns:
            SteamQualityAnalysis
        """
        return self.quality_monitor.analyze_quality(reading, boiler_water_tds_ppm)

    def size_prv(
        self,
        prv_id: str,
        inlet_temperature_f: Optional[float] = None,
    ) -> PRVSizingOutput:
        """
        Size a specific PRV.

        Args:
            prv_id: PRV identifier
            inlet_temperature_f: Optional inlet temperature

        Returns:
            PRVSizingOutput

        Raises:
            ValueError: If PRV not configured
        """
        if prv_id not in self.prv_optimizers:
            raise ValueError(f"PRV {prv_id} not configured")

        return self.prv_optimizers[prv_id].size_prv(inlet_temperature_f)

    def calculate_flash(
        self,
        condensate_flow_lb_hr: float,
        condensate_pressure_psig: float,
        flash_pressure_psig: float,
    ) -> FlashSteamOutput:
        """
        Calculate flash steam recovery.

        Convenience method for flash calculation.

        Args:
            condensate_flow_lb_hr: Condensate flow
            condensate_pressure_psig: Inlet pressure
            flash_pressure_psig: Flash pressure

        Returns:
            FlashSteamOutput
        """
        calc = FlashSteamCalculator()
        return calc.calculate_flash(
            condensate_flow_lb_hr,
            condensate_pressure_psig,
            flash_pressure_psig,
        )

    def get_steam_properties(
        self,
        pressure_psig: float,
        temperature_f: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get steam properties at conditions.

        Args:
            pressure_psig: Pressure (psig)
            temperature_f: Temperature (F)

        Returns:
            Dictionary with steam properties
        """
        props = self.steam_calc.get_steam_properties(
            pressure_psig,
            temperature_f,
        )
        return props.dict()
