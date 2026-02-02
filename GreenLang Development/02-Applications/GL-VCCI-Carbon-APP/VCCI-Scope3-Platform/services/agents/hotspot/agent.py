# -*- coding: utf-8 -*-
"""
HotspotAnalysisAgent
GL-VCCI Scope 3 Platform

Main agent for emissions hotspot analysis and scenario modeling.
Orchestrates Pareto analysis, segmentation, ROI calculation, and insight generation.

Version: 2.0.0 - Enhanced with GreenLang SDK
Phase: 5 (Agent Architecture Compliance)
Date: 2025-11-09
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

# GreenLang SDK Integration
from greenlang.sdk.base import Agent, Metadata, Result
from greenlang.cache import CacheManager, get_cache_manager
from greenlang.telemetry import (
    MetricsCollector,
    get_logger,
    track_execution,
    create_span,
)

from .models import (
    EmissionRecord,
    ParetoAnalysis,
    SegmentationAnalysis,
    HotspotReport,
    InsightReport,
    ScenarioResult,
    ROIAnalysis,
    AbatementCurve,
    Initiative,
    BaseScenario
)
from .config import (
    HotspotAnalysisConfig,
    AnalysisDimension,
    HotspotCriteria,
    DEFAULT_CONFIG
)
from .analysis import ParetoAnalyzer, SegmentationAnalyzer, TrendAnalyzer
from .scenarios import ScenarioEngine
from .roi import ROICalculator, AbatementCurveGenerator
from .insights import HotspotDetector, RecommendationEngine
from .exceptions import HotspotAnalysisError, InsufficientDataError

logger = get_logger(__name__)


class HotspotAnalysisAgent(Agent[List[Dict[str, Any]], Dict[str, Any]]):
    """
    Emissions Hotspot Analysis Agent.

    Comprehensive analysis agent for identifying emissions hotspots,
    modeling reduction scenarios, and generating actionable insights.

    Features:
    - Pareto analysis (80/20 rule)
    - Multi-dimensional segmentation
    - Scenario modeling framework (stubs for Week 27+)
    - ROI analysis
    - Marginal abatement cost curve (MACC) generation
    - Automated hotspot detection
    - Actionable insight generation

    Performance Target: Analyze 100K records in <10 seconds
    """

    def __init__(self, config: Optional[HotspotAnalysisConfig] = None):
        """
        Initialize HotspotAnalysisAgent.

        Args:
            config: Agent configuration (uses defaults if not provided)
        """
        # Initialize base Agent with metadata
        metadata = Metadata(
            id="hotspot_analysis_agent",
            name="HotspotAnalysisAgent",
            version="2.0.0",
            description="Emissions hotspot analysis and scenario modeling agent",
            tags=["hotspot", "analysis", "pareto", "abatement"],
        )
        super().__init__(metadata)

        self.config = config or DEFAULT_CONFIG

        # Initialize GreenLang infrastructure
        self.cache_manager = get_cache_manager()
        self.metrics = MetricsCollector(namespace="vcci.hotspot")

        # Initialize analyzers
        self.pareto_analyzer = ParetoAnalyzer(self.config.pareto_config)
        self.segmentation_analyzer = SegmentationAnalyzer(self.config.segmentation_config)
        self.trend_analyzer = TrendAnalyzer()

        # Initialize scenario engine
        self.scenario_engine = ScenarioEngine()

        # Initialize ROI tools
        self.roi_calculator = ROICalculator(self.config.roi_config)
        self.abatement_curve_generator = AbatementCurveGenerator(self.config.roi_config)

        # Initialize insight tools
        self.hotspot_detector = HotspotDetector(self.config.hotspot_criteria)
        self.recommendation_engine = RecommendationEngine()

        logger.info(
            "Initialized HotspotAnalysisAgent v2.0 with config: "
            f"max_records={self.config.max_records_in_memory}, "
            f"parallel={self.config.enable_parallel_processing}"
        )

    def validate(self, input_data: List[Dict[str, Any]]) -> bool:
        """
        Validate input emissions data.

        Args:
            input_data: List of emission records

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, list):
            logger.error("Input data must be a list")
            return False

        if not input_data:
            logger.warning("Empty input data")
            return True

        # Validate each record has required fields
        for i, record in enumerate(input_data):
            if not isinstance(record, dict):
                logger.error(f"Record {i} is not a dictionary")
                return False

            if "emissions_tco2e" not in record:
                logger.error(f"Record {i} missing 'emissions_tco2e' field")
                return False

        return True

    @track_execution(metric_name="hotspot_process")
    def process(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process emissions data for comprehensive hotspot analysis.

        Args:
            input_data: List of emission records

        Returns:
            Dictionary with all analysis results
        """
        with create_span(name="hotspot_analysis", attributes={"record_count": len(input_data)}):
            result = self.analyze_comprehensive(input_data)

        # Record metrics
        if self.metrics:
            self.metrics.record_metric(
                "hotspots.total",
                result.get("summary", {}).get("n_hotspots", 0),
                unit="count"
            )

        return result

    # ========================================================================
    # PARETO ANALYSIS
    # ========================================================================

    def analyze_pareto(
        self,
        emissions_data: List[Dict[str, Any]],
        dimension: str = "supplier_name"
    ) -> ParetoAnalysis:
        """
        Perform Pareto analysis (80/20 rule).

        Identifies top 20% of contributors responsible for 80% of emissions.

        Args:
            emissions_data: List of emission records
            dimension: Dimension to analyze (supplier_name, scope3_category, etc.)

        Returns:
            ParetoAnalysis with top contributors

        Raises:
            HotspotAnalysisError: If analysis fails
        """
        try:
            start_time = time.time()

            logger.info(
                f"Starting Pareto analysis: {len(emissions_data)} records, "
                f"dimension={dimension}"
            )

            result = self.pareto_analyzer.analyze(emissions_data, dimension)

            elapsed = time.time() - start_time
            logger.info(f"Pareto analysis completed in {elapsed:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Pareto analysis failed: {e}", exc_info=True)
            raise HotspotAnalysisError(f"Pareto analysis failed: {e}") from e

    # ========================================================================
    # SEGMENTATION ANALYSIS
    # ========================================================================

    def analyze_segmentation(
        self,
        emissions_data: List[Dict[str, Any]],
        dimensions: List[AnalysisDimension] = None
    ) -> Dict[AnalysisDimension, SegmentationAnalysis]:
        """
        Perform multi-dimensional segmentation analysis.

        Segments emissions by supplier, category, product, region, facility, etc.

        Args:
            emissions_data: List of emission records
            dimensions: List of dimensions to analyze (default: all available)

        Returns:
            Dictionary mapping dimension to segmentation analysis

        Raises:
            HotspotAnalysisError: If analysis fails
        """
        try:
            start_time = time.time()

            if dimensions is None:
                dimensions = [
                    AnalysisDimension.SUPPLIER,
                    AnalysisDimension.CATEGORY,
                    AnalysisDimension.PRODUCT,
                    AnalysisDimension.REGION
                ]

            logger.info(
                f"Starting segmentation analysis: {len(emissions_data)} records, "
                f"{len(dimensions)} dimensions"
            )

            results = self.segmentation_analyzer.analyze_multiple_dimensions(
                emissions_data,
                dimensions
            )

            elapsed = time.time() - start_time
            logger.info(
                f"Segmentation analysis completed in {elapsed:.2f}s, "
                f"{len(results)} dimensions analyzed"
            )

            return results

        except Exception as e:
            logger.error(f"Segmentation analysis failed: {e}", exc_info=True)
            raise HotspotAnalysisError(f"Segmentation analysis failed: {e}") from e

    # ========================================================================
    # SCENARIO MODELING
    # ========================================================================

    def model_scenario(
        self,
        scenario: BaseScenario,
        baseline_data: Optional[List[Dict[str, Any]]] = None
    ) -> ScenarioResult:
        """
        Model emission reduction scenario.

        NOTE: This is a framework implementation. Full scenario modeling
        logic will be implemented in Week 27+.

        Args:
            scenario: Scenario configuration
            baseline_data: Baseline emission data for context

        Returns:
            ScenarioResult with projected impact

        Raises:
            HotspotAnalysisError: If modeling fails
        """
        try:
            logger.info(f"Modeling scenario: {scenario.name}")

            result = self.scenario_engine.model_scenario(scenario, baseline_data)

            logger.info(
                f"Scenario modeling complete: {result.reduction_tco2e:.1f} tCO2e reduction"
            )

            return result

        except Exception as e:
            logger.error(f"Scenario modeling failed: {e}", exc_info=True)
            raise HotspotAnalysisError(f"Scenario modeling failed: {e}") from e

    def compare_scenarios(
        self,
        scenarios: List[BaseScenario],
        baseline_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple reduction scenarios.

        Args:
            scenarios: List of scenarios to compare
            baseline_data: Baseline emission data

        Returns:
            Scenario comparison analysis
        """
        return self.scenario_engine.compare_scenarios(scenarios, baseline_data)

    # ========================================================================
    # ROI ANALYSIS
    # ========================================================================

    def calculate_roi(self, initiative: Initiative) -> ROIAnalysis:
        """
        Calculate ROI for emission reduction initiative.

        Includes NPV, IRR, payback period, and carbon value.

        Args:
            initiative: Emission reduction initiative

        Returns:
            ROIAnalysis with comprehensive metrics

        Raises:
            HotspotAnalysisError: If calculation fails
        """
        try:
            logger.info(f"Calculating ROI for initiative: {initiative.name}")

            result = self.roi_calculator.calculate(initiative)

            logger.info(
                f"ROI calculation complete: ${result.roi_usd_per_tco2e:.2f}/tCO2e, "
                f"NPV=${result.npv_10y_usd:,.0f}"
            )

            return result

        except Exception as e:
            logger.error(f"ROI calculation failed: {e}", exc_info=True)
            raise HotspotAnalysisError(f"ROI calculation failed: {e}") from e

    # ========================================================================
    # ABATEMENT CURVE
    # ========================================================================

    def generate_abatement_curve(
        self,
        initiatives: List[Initiative]
    ) -> AbatementCurve:
        """
        Generate marginal abatement cost curve (MACC).

        Visualizes initiatives sorted by cost-effectiveness.

        Args:
            initiatives: List of emission reduction initiatives

        Returns:
            AbatementCurve with sorted initiatives

        Raises:
            HotspotAnalysisError: If generation fails
        """
        try:
            logger.info(f"Generating abatement curve for {len(initiatives)} initiatives")

            result = self.abatement_curve_generator.generate(initiatives)

            logger.info(
                f"Abatement curve generated: {len(result.initiatives)} initiatives, "
                f"total reduction={result.total_reduction_potential_tco2e:,.0f} tCO2e"
            )

            return result

        except Exception as e:
            logger.error(f"Abatement curve generation failed: {e}", exc_info=True)
            raise HotspotAnalysisError(f"Abatement curve generation failed: {e}") from e

    # ========================================================================
    # HOTSPOT DETECTION
    # ========================================================================

    def identify_hotspots(
        self,
        emissions_data: List[Dict[str, Any]],
        criteria: Optional[HotspotCriteria] = None
    ) -> HotspotReport:
        """
        Identify emissions hotspots using configurable criteria.

        Flags high-emission entities, poor data quality, and concentration risks.

        Args:
            emissions_data: List of emission records
            criteria: Hotspot criteria (uses config default if not provided)

        Returns:
            HotspotReport with identified hotspots

        Raises:
            HotspotAnalysisError: If detection fails
        """
        try:
            start_time = time.time()

            if criteria:
                detector = HotspotDetector(criteria)
            else:
                detector = self.hotspot_detector

            logger.info(f"Detecting hotspots in {len(emissions_data)} records")

            result = detector.detect(emissions_data)

            elapsed = time.time() - start_time
            logger.info(
                f"Hotspot detection completed in {elapsed:.2f}s, "
                f"{result.n_hotspots} hotspots found"
            )

            return result

        except Exception as e:
            logger.error(f"Hotspot detection failed: {e}", exc_info=True)
            raise HotspotAnalysisError(f"Hotspot detection failed: {e}") from e

    # ========================================================================
    # INSIGHT GENERATION
    # ========================================================================

    def generate_insights(
        self,
        emissions_data: Optional[List[Dict[str, Any]]] = None,
        hotspot_report: Optional[HotspotReport] = None,
        pareto_analysis: Optional[ParetoAnalysis] = None,
        segmentation_analysis: Optional[SegmentationAnalysis] = None,
        abatement_curve: Optional[AbatementCurve] = None
    ) -> InsightReport:
        """
        Generate actionable insights from analysis results.

        Can accept either raw emissions data (will run analysis) or
        pre-computed analysis results.

        Args:
            emissions_data: Raw emission records (will run full analysis)
            hotspot_report: Pre-computed hotspot report
            pareto_analysis: Pre-computed Pareto analysis
            segmentation_analysis: Pre-computed segmentation
            abatement_curve: Pre-computed abatement curve

        Returns:
            InsightReport with prioritized recommendations

        Raises:
            HotspotAnalysisError: If generation fails
        """
        try:
            logger.info("Generating actionable insights")

            # Run analysis if raw data provided
            if emissions_data and not hotspot_report:
                hotspot_report = self.identify_hotspots(emissions_data)

            if emissions_data and not pareto_analysis:
                pareto_analysis = self.analyze_pareto(emissions_data)

            # Generate insights
            result = self.recommendation_engine.generate_insights(
                hotspot_report=hotspot_report,
                pareto_analysis=pareto_analysis,
                segmentation_analysis=segmentation_analysis,
                abatement_curve=abatement_curve
            )

            logger.info(
                f"Generated {result.total_insights} insights: "
                f"{len(result.critical_insights)} critical, "
                f"{len(result.high_insights)} high priority"
            )

            return result

        except Exception as e:
            logger.error(f"Insight generation failed: {e}", exc_info=True)
            raise HotspotAnalysisError(f"Insight generation failed: {e}") from e

    # ========================================================================
    # COMPREHENSIVE ANALYSIS
    # ========================================================================

    def analyze_comprehensive(
        self,
        emissions_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis including all available analyses.

        This is the primary analysis method that runs:
        - Pareto analysis
        - Multi-dimensional segmentation
        - Hotspot detection
        - Insight generation

        Args:
            emissions_data: List of emission records

        Returns:
            Dictionary with all analysis results

        Raises:
            HotspotAnalysisError: If analysis fails
        """
        try:
            start_time = time.time()

            logger.info(
                f"Starting comprehensive analysis on {len(emissions_data)} records"
            )

            # Validate data
            if not emissions_data:
                raise InsufficientDataError("No emission data provided")

            # Run analyses
            results = {}

            # Pareto analysis
            try:
                results["pareto"] = self.analyze_pareto(emissions_data)
            except Exception as e:
                logger.warning(f"Pareto analysis failed: {e}")
                results["pareto"] = None

            # Segmentation analysis
            try:
                results["segmentation"] = self.analyze_segmentation(emissions_data)
            except Exception as e:
                logger.warning(f"Segmentation analysis failed: {e}")
                results["segmentation"] = None

            # Hotspot detection
            try:
                results["hotspots"] = self.identify_hotspots(emissions_data)
            except Exception as e:
                logger.warning(f"Hotspot detection failed: {e}")
                results["hotspots"] = None

            # Generate insights
            try:
                results["insights"] = self.generate_insights(
                    hotspot_report=results.get("hotspots"),
                    pareto_analysis=results.get("pareto"),
                    segmentation_analysis=results.get("segmentation", {}).get(
                        AnalysisDimension.SUPPLIER
                    )
                )
            except Exception as e:
                logger.warning(f"Insight generation failed: {e}")
                results["insights"] = None

            # Summary statistics
            total_emissions = sum(r.get("emissions_tco2e", 0) for r in emissions_data)
            results["summary"] = {
                "total_records": len(emissions_data),
                "total_emissions_tco2e": round(total_emissions, 2),
                "n_hotspots": results["hotspots"].n_hotspots if results.get("hotspots") else 0,
                "n_insights": results["insights"].total_insights if results.get("insights") else 0,
                "processing_time_seconds": time.time() - start_time
            }

            elapsed = time.time() - start_time
            logger.info(
                f"Comprehensive analysis completed in {elapsed:.2f}s: "
                f"{results['summary']['n_hotspots']} hotspots, "
                f"{results['summary']['n_insights']} insights"
            )

            return results

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}", exc_info=True)
            raise HotspotAnalysisError(f"Comprehensive analysis failed: {e}") from e


__all__ = ["HotspotAnalysisAgent"]
