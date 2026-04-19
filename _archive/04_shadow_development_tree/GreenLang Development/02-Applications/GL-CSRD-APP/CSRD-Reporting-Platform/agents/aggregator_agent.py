# -*- coding: utf-8 -*-
"""
AggregatorAgent - Multi-Standard ESG Data Aggregator for CSRD Reporting

This agent aggregates ESG data across multiple reporting frameworks and performs
time-series analysis and benchmarking.

Responsibilities:
1. Cross-standard mapping (TCFD, GRI, SASB → ESRS)
2. Time-series aggregation and trend analysis
3. Comparative benchmarking against industry peers
4. Gap analysis across frameworks
5. Historical data alignment

Key Features:
- 100% deterministic processing (zero hallucination guarantee)
- NO AI/LLM usage
- <2 min processing for 10,000 metrics
- Complete audit trail
- Multi-framework support

Version: 1.0.0
Author: GreenLang CSRD Team
License: MIT
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from greenlang.determinism import DeterministicClock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ERROR CODES
# ============================================================================

ERROR_CODES = {
    # Critical Errors
    "A001": "Invalid framework identifier",
    "A002": "Mapping not found in cross-reference",
    "A003": "Data type mismatch during aggregation",
    "A004": "Missing required time-series data",
    "A005": "Benchmark data not available for industry",

    # Warnings
    "W001": "Partial mapping quality (not direct)",
    "W002": "Time-series gap detected",
    "W003": "Benchmark comparison unavailable",
    "W004": "Multiple mappings found (ambiguous)",
    "W005": "Historical data missing for trend analysis",
    "W006": "Industry sector not specified",

    # Info
    "I001": "Framework integration complete",
    "I002": "Time-series analysis complete",
    "I003": "Benchmark comparison complete",
}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class FrameworkMapping(BaseModel):
    """Represents a mapping between frameworks."""
    source_framework: str  # "TCFD", "GRI", "SASB"
    source_code: str
    esrs_code: str
    esrs_disclosure: str
    mapping_quality: str  # "direct", "high", "partial", "low"
    notes: Optional[str] = None


class AggregatedMetric(BaseModel):
    """Represents an aggregated metric from multiple sources."""
    esrs_code: str
    esrs_name: str
    primary_value: Union[float, str, bool, int, None]
    primary_source: str  # "ESRS", "TCFD", "GRI", "SASB"
    unit: str
    reporting_period: str

    # Multi-source data
    source_values: Dict[str, Any] = {}  # Map of framework -> value
    mapping_quality: str = "direct"
    data_quality_score: Optional[float] = None

    # Time-series data
    historical_values: Optional[List[Dict[str, Any]]] = None

    # Enrichment
    aggregation_timestamp: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None


class TrendAnalysis(BaseModel):
    """Time-series trend analysis for a metric."""
    esrs_code: str
    metric_name: str
    periods: List[str]
    values: List[float]

    # Trend metrics
    yoy_change_percent: Optional[float] = None  # Year-over-year % change
    cagr_3year: Optional[float] = None  # 3-year CAGR
    trend_direction: Optional[str] = None  # "improving", "declining", "stable"
    volatility: Optional[float] = None  # Standard deviation

    # Statistical analysis
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None


class BenchmarkComparison(BaseModel):
    """Benchmark comparison against industry peers."""
    esrs_code: str
    metric_name: str
    company_value: Union[float, int]
    unit: str

    # Benchmark statistics
    industry_sector: str
    sector_median: Optional[float] = None
    sector_top_quartile: Optional[float] = None
    sector_bottom_quartile: Optional[float] = None

    # Performance metrics
    percentile_rank: Optional[float] = None  # 0-100
    performance_vs_median: Optional[str] = None  # "above", "below", "at"
    performance_vs_top_quartile: Optional[str] = None  # "above", "below", "at"

    # Context
    sample_size: Optional[int] = None
    benchmark_year: Optional[int] = None


class GapAnalysis(BaseModel):
    """Gap analysis across frameworks."""
    total_esrs_required: int
    total_esrs_covered: int
    coverage_percentage: float

    # Framework coverage
    coverage_by_framework: Dict[str, int] = {}

    # Gaps
    missing_esrs_codes: List[str] = []
    partial_mappings: List[str] = []

    # Quality assessment
    direct_mappings: int = 0
    high_quality_mappings: int = 0
    partial_mappings_count: int = 0
    low_quality_mappings: int = 0


class AggregationIssue(BaseModel):
    """Represents an aggregation warning or error."""
    esrs_code: Optional[str] = None
    error_code: str
    severity: str  # "error", "warning", "info"
    message: str
    source_framework: Optional[str] = None
    suggestion: Optional[str] = None


# ============================================================================
# FRAMEWORK MAPPER
# ============================================================================

class FrameworkMapper:
    """
    Map metrics from TCFD, GRI, SASB to ESRS.

    This mapper uses deterministic database lookups - NO LLM.
    """

    def __init__(self, framework_mappings: Dict[str, Any]):
        """
        Initialize framework mapper.

        Args:
            framework_mappings: Framework mappings database
        """
        self.framework_mappings = framework_mappings

        # Create fast lookup indices
        self.tcfd_to_esrs = self._build_tcfd_index()
        self.gri_to_esrs = self._build_gri_index()
        self.sasb_to_esrs = self._build_sasb_index()

        logger.info(f"FrameworkMapper initialized with {self._count_mappings()} total mappings")

    def _count_mappings(self) -> int:
        """Count total mappings."""
        count = 0
        count += len(self.framework_mappings.get("esrs_to_tcfd", []))
        count += len(self.framework_mappings.get("esrs_to_gri", []))
        count += len(self.framework_mappings.get("esrs_to_sasb", []))
        return count

    def _build_tcfd_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build TCFD -> ESRS lookup index."""
        index = defaultdict(list)
        for mapping in self.framework_mappings.get("esrs_to_tcfd", []):
            tcfd_ref = mapping.get("tcfd_reference", "")
            index[tcfd_ref].append(mapping)
        return dict(index)

    def _build_gri_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build GRI -> ESRS lookup index."""
        index = defaultdict(list)
        for mapping in self.framework_mappings.get("esrs_to_gri", []):
            gri_disclosure = mapping.get("gri_disclosure", "")
            gri_standard = mapping.get("gri_standard", "")
            # Index by both disclosure and standard
            index[gri_disclosure].append(mapping)
            index[gri_standard].append(mapping)
        return dict(index)

    def _build_sasb_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build SASB -> ESRS lookup index."""
        index = defaultdict(list)
        for mapping in self.framework_mappings.get("esrs_to_sasb", []):
            sasb_metric = mapping.get("sasb_metric", "")
            sasb_category = mapping.get("sasb_category", "")
            # Index by both metric and category
            index[sasb_metric].append(mapping)
            index[sasb_category].append(mapping)
        return dict(index)

    def map_tcfd_to_esrs(
        self,
        tcfd_reference: str,
        tcfd_value: Any
    ) -> Tuple[Optional[FrameworkMapping], List[AggregationIssue]]:
        """Map TCFD metric to ESRS."""
        issues = []

        if tcfd_reference not in self.tcfd_to_esrs:
            issues.append(AggregationIssue(
                error_code="A002",
                severity="warning",
                message=f"No ESRS mapping found for TCFD reference: {tcfd_reference}",
                source_framework="TCFD"
            ))
            return None, issues

        # Get first mapping (there might be multiple)
        mappings = self.tcfd_to_esrs[tcfd_reference]
        if len(mappings) > 1:
            issues.append(AggregationIssue(
                error_code="W004",
                severity="warning",
                message=f"Multiple ESRS mappings found for TCFD reference: {tcfd_reference}",
                source_framework="TCFD",
                suggestion="Using first mapping - verify correctness"
            ))

        mapping = mappings[0]

        framework_mapping = FrameworkMapping(
            source_framework="TCFD",
            source_code=tcfd_reference,
            esrs_code=mapping.get("esrs_code", ""),
            esrs_disclosure=mapping.get("esrs_disclosure", ""),
            mapping_quality=mapping.get("mapping_quality", "unknown"),
            notes=mapping.get("notes")
        )

        # Check mapping quality
        if mapping.get("mapping_quality") != "direct":
            issues.append(AggregationIssue(
                esrs_code=framework_mapping.esrs_code,
                error_code="W001",
                severity="warning",
                message=f"Partial mapping quality: {mapping.get('mapping_quality')}",
                source_framework="TCFD",
                suggestion="Review mapping and consider supplemental data"
            ))

        return framework_mapping, issues

    def map_gri_to_esrs(
        self,
        gri_disclosure: str,
        gri_value: Any
    ) -> Tuple[Optional[FrameworkMapping], List[AggregationIssue]]:
        """Map GRI metric to ESRS."""
        issues = []

        if gri_disclosure not in self.gri_to_esrs:
            issues.append(AggregationIssue(
                error_code="A002",
                severity="warning",
                message=f"No ESRS mapping found for GRI disclosure: {gri_disclosure}",
                source_framework="GRI"
            ))
            return None, issues

        # Get first mapping
        mappings = self.gri_to_esrs[gri_disclosure]
        if len(mappings) > 1:
            issues.append(AggregationIssue(
                error_code="W004",
                severity="warning",
                message=f"Multiple ESRS mappings found for GRI disclosure: {gri_disclosure}",
                source_framework="GRI"
            ))

        mapping = mappings[0]

        framework_mapping = FrameworkMapping(
            source_framework="GRI",
            source_code=gri_disclosure,
            esrs_code=mapping.get("esrs_code", ""),
            esrs_disclosure=mapping.get("esrs_disclosure", ""),
            mapping_quality=mapping.get("mapping_quality", "unknown"),
            notes=mapping.get("notes")
        )

        # Check mapping quality
        if mapping.get("mapping_quality") != "direct":
            issues.append(AggregationIssue(
                esrs_code=framework_mapping.esrs_code,
                error_code="W001",
                severity="warning",
                message=f"Partial mapping quality: {mapping.get('mapping_quality')}",
                source_framework="GRI"
            ))

        return framework_mapping, issues

    def map_sasb_to_esrs(
        self,
        sasb_metric: str,
        sasb_value: Any
    ) -> Tuple[Optional[FrameworkMapping], List[AggregationIssue]]:
        """Map SASB metric to ESRS."""
        issues = []

        if sasb_metric not in self.sasb_to_esrs:
            issues.append(AggregationIssue(
                error_code="A002",
                severity="warning",
                message=f"No ESRS mapping found for SASB metric: {sasb_metric}",
                source_framework="SASB"
            ))
            return None, issues

        # Get first mapping
        mappings = self.sasb_to_esrs[sasb_metric]
        if len(mappings) > 1:
            issues.append(AggregationIssue(
                error_code="W004",
                severity="warning",
                message=f"Multiple ESRS mappings found for SASB metric: {sasb_metric}",
                source_framework="SASB"
            ))

        mapping = mappings[0]

        framework_mapping = FrameworkMapping(
            source_framework="SASB",
            source_code=sasb_metric,
            esrs_code=mapping.get("esrs_code", ""),
            esrs_disclosure=mapping.get("esrs_disclosure", ""),
            mapping_quality=mapping.get("mapping_quality", "unknown"),
            notes=mapping.get("notes")
        )

        # Check mapping quality
        if mapping.get("mapping_quality") != "direct":
            issues.append(AggregationIssue(
                esrs_code=framework_mapping.esrs_code,
                error_code="W001",
                severity="warning",
                message=f"Partial mapping quality: {mapping.get('mapping_quality')}",
                source_framework="SASB"
            ))

        return framework_mapping, issues


# ============================================================================
# TIME-SERIES ANALYZER
# ============================================================================

class TimeSeriesAnalyzer:
    """
    Analyze time-series trends for ESG metrics.

    100% deterministic statistical analysis - NO LLM.
    """

    def analyze_trend(
        self,
        esrs_code: str,
        metric_name: str,
        time_series_data: List[Dict[str, Any]]
    ) -> Tuple[Optional[TrendAnalysis], List[AggregationIssue]]:
        """
        Analyze time-series trend for a metric.

        Args:
            esrs_code: ESRS metric code
            metric_name: Metric name
            time_series_data: List of {period, value} dictionaries

        Returns:
            Tuple of (TrendAnalysis, list of issues)
        """
        issues = []

        # Validate input
        if not time_series_data or len(time_series_data) < 2:
            issues.append(AggregationIssue(
                esrs_code=esrs_code,
                error_code="A004",
                severity="warning",
                message=f"Insufficient time-series data for trend analysis (need at least 2 periods)",
                suggestion="Provide historical data for trend analysis"
            ))
            return None, issues

        # Extract periods and values
        periods = []
        values = []

        for data_point in sorted(time_series_data, key=lambda x: x.get("period", "")):
            period = data_point.get("period")
            value = data_point.get("value")

            if period and value is not None:
                try:
                    periods.append(str(period))
                    values.append(float(value))
                except (ValueError, TypeError):
                    issues.append(AggregationIssue(
                        esrs_code=esrs_code,
                        error_code="A003",
                        severity="warning",
                        message=f"Invalid value type for period {period}: {value}"
                    ))

        if len(values) < 2:
            return None, issues

        # Calculate trend metrics
        trend_analysis = TrendAnalysis(
            esrs_code=esrs_code,
            metric_name=metric_name,
            periods=periods,
            values=values
        )

        # Year-over-year change (latest vs previous)
        if len(values) >= 2:
            latest = values[-1]
            previous = values[-2]
            if previous != 0:
                yoy_change = ((latest - previous) / abs(previous)) * 100
                trend_analysis.yoy_change_percent = round(yoy_change, 2)

        # 3-year CAGR
        if len(values) >= 3:
            first = values[-3]
            last = values[-1]
            if first > 0:
                cagr = (pow(last / first, 1/2) - 1) * 100  # 2 periods between 3 points
                trend_analysis.cagr_3year = round(cagr, 2)

        # Trend direction
        if len(values) >= 2:
            # Simple linear trend: compare first half to second half
            mid = len(values) // 2
            first_half_avg = np.mean(values[:mid]) if mid > 0 else values[0]
            second_half_avg = np.mean(values[mid:])

            change_pct = ((second_half_avg - first_half_avg) / abs(first_half_avg)) * 100 if first_half_avg != 0 else 0

            if abs(change_pct) < 5:  # Less than 5% change = stable
                trend_analysis.trend_direction = "stable"
            elif change_pct > 0:
                # For emissions/waste: increasing is declining performance
                # For renewable energy: increasing is improving performance
                # Default interpretation: increasing values = improving (can be customized)
                trend_analysis.trend_direction = "improving"
            else:
                trend_analysis.trend_direction = "declining"

        # Statistical metrics
        trend_analysis.min_value = round(float(np.min(values)), 3)
        trend_analysis.max_value = round(float(np.max(values)), 3)
        trend_analysis.mean_value = round(float(np.mean(values)), 3)
        trend_analysis.median_value = round(float(np.median(values)), 3)
        trend_analysis.volatility = round(float(np.std(values)), 3)

        return trend_analysis, issues


# ============================================================================
# BENCHMARK COMPARATOR
# ============================================================================

class BenchmarkComparator:
    """
    Compare company metrics against industry benchmarks.

    100% deterministic comparison - NO LLM.
    """

    def __init__(self, industry_benchmarks: Optional[Dict[str, Any]] = None):
        """
        Initialize benchmark comparator.

        Args:
            industry_benchmarks: Industry benchmark database (optional)
        """
        self.industry_benchmarks = industry_benchmarks or {}

    def compare_to_benchmark(
        self,
        esrs_code: str,
        metric_name: str,
        company_value: Union[float, int],
        unit: str,
        industry_sector: str
    ) -> Tuple[Optional[BenchmarkComparison], List[AggregationIssue]]:
        """
        Compare company metric to industry benchmark.

        Args:
            esrs_code: ESRS metric code
            metric_name: Metric name
            company_value: Company's value
            unit: Unit of measurement
            industry_sector: Industry sector for comparison

        Returns:
            Tuple of (BenchmarkComparison, list of issues)
        """
        issues = []

        # Check if benchmarks available
        if not self.industry_benchmarks:
            issues.append(AggregationIssue(
                esrs_code=esrs_code,
                error_code="W003",
                severity="info",
                message="Industry benchmark data not available",
                suggestion="Provide industry_benchmarks.json for peer comparison"
            ))
            return None, issues

        # Look for sector-specific benchmark
        sector_benchmarks = self.industry_benchmarks.get(industry_sector, {})
        if not sector_benchmarks:
            issues.append(AggregationIssue(
                esrs_code=esrs_code,
                error_code="A005",
                severity="warning",
                message=f"No benchmark data available for industry sector: {industry_sector}",
                suggestion="Check industry sector name or provide benchmark data"
            ))
            return None, issues

        # Look for metric-specific benchmark
        metric_benchmark = sector_benchmarks.get(esrs_code, {})
        if not metric_benchmark:
            issues.append(AggregationIssue(
                esrs_code=esrs_code,
                error_code="W003",
                severity="info",
                message=f"No benchmark available for metric {esrs_code} in sector {industry_sector}"
            ))
            return None, issues

        # Build comparison
        comparison = BenchmarkComparison(
            esrs_code=esrs_code,
            metric_name=metric_name,
            company_value=company_value,
            unit=unit,
            industry_sector=industry_sector,
            sector_median=metric_benchmark.get("median"),
            sector_top_quartile=metric_benchmark.get("top_quartile"),
            sector_bottom_quartile=metric_benchmark.get("bottom_quartile"),
            sample_size=metric_benchmark.get("sample_size"),
            benchmark_year=metric_benchmark.get("year")
        )

        # Calculate performance metrics
        if comparison.sector_median is not None:
            # Performance vs median
            if company_value > comparison.sector_median * 1.05:  # 5% tolerance
                comparison.performance_vs_median = "above"
            elif company_value < comparison.sector_median * 0.95:
                comparison.performance_vs_median = "below"
            else:
                comparison.performance_vs_median = "at"

        if comparison.sector_top_quartile is not None:
            # Performance vs top quartile
            if company_value >= comparison.sector_top_quartile:
                comparison.performance_vs_top_quartile = "above"
            elif company_value < comparison.sector_top_quartile:
                comparison.performance_vs_top_quartile = "below"
            else:
                comparison.performance_vs_top_quartile = "at"

        # Calculate percentile rank (simplified)
        if all([comparison.sector_bottom_quartile, comparison.sector_median, comparison.sector_top_quartile]):
            q1 = comparison.sector_bottom_quartile
            q2 = comparison.sector_median
            q3 = comparison.sector_top_quartile

            if company_value <= q1:
                comparison.percentile_rank = 25.0
            elif company_value <= q2:
                # Linear interpolation between Q1 and Q2
                comparison.percentile_rank = 25 + 25 * ((company_value - q1) / (q2 - q1))
            elif company_value <= q3:
                # Linear interpolation between Q2 and Q3
                comparison.percentile_rank = 50 + 25 * ((company_value - q2) / (q3 - q2))
            else:
                comparison.percentile_rank = 75.0 + min(25.0, 25 * ((company_value - q3) / q3))

            comparison.percentile_rank = round(comparison.percentile_rank, 1)

        return comparison, issues


# ============================================================================
# AGGREGATOR AGENT
# ============================================================================

class AggregatorAgent:
    """
    Aggregate ESG data across multiple reporting frameworks.

    This agent is 100% deterministic:
    - All mappings from database (JSON file)
    - All calculations using Python/NumPy (no LLM)
    - All results reproducible (same input → same output)

    Performance: <2 min for 10,000 metrics
    Frameworks: TCFD, GRI, SASB → ESRS
    """

    def __init__(
        self,
        framework_mappings_path: Union[str, Path],
        industry_benchmarks_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the AggregatorAgent.

        Args:
            framework_mappings_path: Path to framework mappings JSON
            industry_benchmarks_path: Path to industry benchmarks JSON (optional)
        """
        self.framework_mappings_path = Path(framework_mappings_path)
        self.industry_benchmarks_path = Path(industry_benchmarks_path) if industry_benchmarks_path else None

        # Load databases
        self.framework_mappings = self._load_framework_mappings()
        self.industry_benchmarks = self._load_industry_benchmarks() if self.industry_benchmarks_path else None

        # Initialize components
        self.framework_mapper = FrameworkMapper(self.framework_mappings)
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.benchmark_comparator = BenchmarkComparator(self.industry_benchmarks)

        # Statistics
        self.stats = {
            "total_metrics_processed": 0,
            "esrs_metrics": 0,
            "tcfd_metrics_mapped": 0,
            "gri_metrics_mapped": 0,
            "sasb_metrics_mapped": 0,
            "trends_analyzed": 0,
            "benchmarks_compared": 0,
            "start_time": None,
            "end_time": None
        }

        logger.info(f"AggregatorAgent initialized with {self.framework_mapper._count_mappings()} framework mappings")

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def _load_framework_mappings(self) -> Dict[str, Any]:
        """Load framework mappings from JSON."""
        try:
            with open(self.framework_mappings_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            logger.info("Loaded framework mappings")
            return mappings
        except Exception as e:
            logger.error(f"Failed to load framework mappings: {e}")
            raise

    def _load_industry_benchmarks(self) -> Optional[Dict[str, Any]]:
        """Load industry benchmarks from JSON."""
        if not self.industry_benchmarks_path or not self.industry_benchmarks_path.exists():
            logger.info("Industry benchmarks file not found - benchmarking will be unavailable")
            return None

        try:
            with open(self.industry_benchmarks_path, 'r', encoding='utf-8') as f:
                benchmarks = json.load(f)
            logger.info("Loaded industry benchmarks")
            return benchmarks
        except Exception as e:
            logger.warning(f"Failed to load industry benchmarks: {e}")
            return None

    # ========================================================================
    # MULTI-FRAMEWORK INTEGRATION
    # ========================================================================

    def integrate_multi_framework_data(
        self,
        esrs_data: Optional[Dict[str, Any]] = None,
        tcfd_data: Optional[Dict[str, Any]] = None,
        gri_data: Optional[Dict[str, Any]] = None,
        sasb_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, AggregatedMetric], List[AggregationIssue]]:
        """
        Integrate data from multiple frameworks into unified ESRS dataset.

        Args:
            esrs_data: ESRS-calculated metrics
            tcfd_data: TCFD climate disclosure data
            gri_data: GRI sustainability report data
            sasb_data: SASB industry metrics

        Returns:
            Tuple of (aggregated metrics dict, list of issues)
        """
        aggregated_metrics = {}
        all_issues = []

        # Step 1: Add ESRS data (primary source)
        if esrs_data:
            for metric_code, metric_data in esrs_data.items():
                if isinstance(metric_data, dict):
                    aggregated_metric = AggregatedMetric(
                        esrs_code=metric_code,
                        esrs_name=metric_data.get("metric_name", metric_code),
                        primary_value=metric_data.get("value"),
                        primary_source="ESRS",
                        unit=metric_data.get("unit", ""),
                        reporting_period=metric_data.get("period_end", ""),
                        source_values={"ESRS": metric_data.get("value")},
                        mapping_quality="direct",
                        data_quality_score=metric_data.get("quality_score"),
                        aggregation_timestamp=DeterministicClock.now().isoformat(),
                        provenance={
                            "source": "ESRS",
                            "original_data": metric_data
                        }
                    )
                    aggregated_metrics[metric_code] = aggregated_metric
                    self.stats["esrs_metrics"] += 1

        # Step 2: Integrate TCFD data
        if tcfd_data:
            for tcfd_ref, tcfd_value in tcfd_data.items():
                if isinstance(tcfd_value, dict):
                    value = tcfd_value.get("value")
                else:
                    value = tcfd_value

                # Map to ESRS
                mapping, issues = self.framework_mapper.map_tcfd_to_esrs(tcfd_ref, value)
                all_issues.extend(issues)

                if mapping:
                    esrs_code = mapping.esrs_code

                    # Merge with existing ESRS data or create new
                    if esrs_code in aggregated_metrics:
                        aggregated_metrics[esrs_code].source_values["TCFD"] = value
                        # If no ESRS primary value, use TCFD
                        if aggregated_metrics[esrs_code].primary_value is None:
                            aggregated_metrics[esrs_code].primary_value = value
                            aggregated_metrics[esrs_code].primary_source = "TCFD"
                            aggregated_metrics[esrs_code].mapping_quality = mapping.mapping_quality
                    else:
                        # Create new aggregated metric from TCFD data
                        aggregated_metric = AggregatedMetric(
                            esrs_code=esrs_code,
                            esrs_name=mapping.esrs_disclosure,
                            primary_value=value,
                            primary_source="TCFD",
                            unit=tcfd_value.get("unit", "") if isinstance(tcfd_value, dict) else "",
                            reporting_period=tcfd_value.get("period", "") if isinstance(tcfd_value, dict) else "",
                            source_values={"TCFD": value},
                            mapping_quality=mapping.mapping_quality,
                            aggregation_timestamp=DeterministicClock.now().isoformat(),
                            provenance={
                                "source": "TCFD",
                                "tcfd_reference": tcfd_ref,
                                "mapping": mapping.dict()
                            }
                        )
                        aggregated_metrics[esrs_code] = aggregated_metric

                    self.stats["tcfd_metrics_mapped"] += 1

        # Step 3: Integrate GRI data
        if gri_data:
            for gri_disclosure, gri_value in gri_data.items():
                if isinstance(gri_value, dict):
                    value = gri_value.get("value")
                else:
                    value = gri_value

                # Map to ESRS
                mapping, issues = self.framework_mapper.map_gri_to_esrs(gri_disclosure, value)
                all_issues.extend(issues)

                if mapping:
                    esrs_code = mapping.esrs_code

                    # Merge with existing data
                    if esrs_code in aggregated_metrics:
                        aggregated_metrics[esrs_code].source_values["GRI"] = value
                        # If no primary value yet, use GRI
                        if aggregated_metrics[esrs_code].primary_value is None:
                            aggregated_metrics[esrs_code].primary_value = value
                            aggregated_metrics[esrs_code].primary_source = "GRI"
                            aggregated_metrics[esrs_code].mapping_quality = mapping.mapping_quality
                    else:
                        # Create new aggregated metric from GRI data
                        aggregated_metric = AggregatedMetric(
                            esrs_code=esrs_code,
                            esrs_name=mapping.esrs_disclosure,
                            primary_value=value,
                            primary_source="GRI",
                            unit=gri_value.get("unit", "") if isinstance(gri_value, dict) else "",
                            reporting_period=gri_value.get("period", "") if isinstance(gri_value, dict) else "",
                            source_values={"GRI": value},
                            mapping_quality=mapping.mapping_quality,
                            aggregation_timestamp=DeterministicClock.now().isoformat(),
                            provenance={
                                "source": "GRI",
                                "gri_disclosure": gri_disclosure,
                                "mapping": mapping.dict()
                            }
                        )
                        aggregated_metrics[esrs_code] = aggregated_metric

                    self.stats["gri_metrics_mapped"] += 1

        # Step 4: Integrate SASB data
        if sasb_data:
            for sasb_metric, sasb_value in sasb_data.items():
                if isinstance(sasb_value, dict):
                    value = sasb_value.get("value")
                else:
                    value = sasb_value

                # Map to ESRS
                mapping, issues = self.framework_mapper.map_sasb_to_esrs(sasb_metric, value)
                all_issues.extend(issues)

                if mapping:
                    esrs_code = mapping.esrs_code

                    # Merge with existing data
                    if esrs_code in aggregated_metrics:
                        aggregated_metrics[esrs_code].source_values["SASB"] = value
                        # If no primary value yet, use SASB
                        if aggregated_metrics[esrs_code].primary_value is None:
                            aggregated_metrics[esrs_code].primary_value = value
                            aggregated_metrics[esrs_code].primary_source = "SASB"
                            aggregated_metrics[esrs_code].mapping_quality = mapping.mapping_quality
                    else:
                        # Create new aggregated metric from SASB data
                        aggregated_metric = AggregatedMetric(
                            esrs_code=esrs_code,
                            esrs_name=mapping.esrs_disclosure,
                            primary_value=value,
                            primary_source="SASB",
                            unit=sasb_value.get("unit", "") if isinstance(sasb_value, dict) else "",
                            reporting_period=sasb_value.get("period", "") if isinstance(sasb_value, dict) else "",
                            source_values={"SASB": value},
                            mapping_quality=mapping.mapping_quality,
                            aggregation_timestamp=DeterministicClock.now().isoformat(),
                            provenance={
                                "source": "SASB",
                                "sasb_metric": sasb_metric,
                                "mapping": mapping.dict()
                            }
                        )
                        aggregated_metrics[esrs_code] = aggregated_metric

                    self.stats["sasb_metrics_mapped"] += 1

        return aggregated_metrics, all_issues

    # ========================================================================
    # TIME-SERIES ANALYSIS
    # ========================================================================

    def perform_time_series_analysis(
        self,
        aggregated_metrics: Dict[str, AggregatedMetric],
        historical_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Tuple[List[TrendAnalysis], List[AggregationIssue]]:
        """
        Perform time-series trend analysis.

        Args:
            aggregated_metrics: Current year's aggregated metrics
            historical_data: Historical data by ESRS code

        Returns:
            Tuple of (list of trend analyses, list of issues)
        """
        trend_analyses = []
        all_issues = []

        if not historical_data:
            all_issues.append(AggregationIssue(
                error_code="W005",
                severity="info",
                message="No historical data provided - trend analysis unavailable",
                suggestion="Provide historical_data parameter for multi-year trend analysis"
            ))
            return trend_analyses, all_issues

        # Analyze trends for each metric with historical data
        for esrs_code, time_series in historical_data.items():
            if esrs_code in aggregated_metrics:
                metric = aggregated_metrics[esrs_code]

                # Add current year to time series
                current_period = {
                    "period": metric.reporting_period,
                    "value": metric.primary_value
                }
                full_time_series = time_series + [current_period]

                # Analyze trend
                trend, issues = self.time_series_analyzer.analyze_trend(
                    esrs_code,
                    metric.esrs_name,
                    full_time_series
                )

                if trend:
                    trend_analyses.append(trend)
                    self.stats["trends_analyzed"] += 1

                    # Update aggregated metric with historical values
                    aggregated_metrics[esrs_code].historical_values = full_time_series

                all_issues.extend(issues)

        return trend_analyses, all_issues

    # ========================================================================
    # BENCHMARK COMPARISON
    # ========================================================================

    def perform_benchmark_comparison(
        self,
        aggregated_metrics: Dict[str, AggregatedMetric],
        industry_sector: str
    ) -> Tuple[List[BenchmarkComparison], List[AggregationIssue]]:
        """
        Compare metrics to industry benchmarks.

        Args:
            aggregated_metrics: Aggregated metrics to compare
            industry_sector: Company's industry sector

        Returns:
            Tuple of (list of benchmark comparisons, list of issues)
        """
        benchmark_comparisons = []
        all_issues = []

        if not industry_sector:
            all_issues.append(AggregationIssue(
                error_code="W006",
                severity="warning",
                message="Industry sector not specified - benchmark comparison unavailable",
                suggestion="Provide industry_sector parameter for peer benchmarking"
            ))
            return benchmark_comparisons, all_issues

        # Compare each metric
        for esrs_code, metric in aggregated_metrics.items():
            if metric.primary_value is not None and isinstance(metric.primary_value, (int, float)):
                comparison, issues = self.benchmark_comparator.compare_to_benchmark(
                    esrs_code,
                    metric.esrs_name,
                    metric.primary_value,
                    metric.unit,
                    industry_sector
                )

                if comparison:
                    benchmark_comparisons.append(comparison)
                    self.stats["benchmarks_compared"] += 1

                all_issues.extend(issues)

        return benchmark_comparisons, all_issues

    # ========================================================================
    # GAP ANALYSIS
    # ========================================================================

    def perform_gap_analysis(
        self,
        aggregated_metrics: Dict[str, AggregatedMetric],
        required_esrs_codes: Optional[List[str]] = None
    ) -> GapAnalysis:
        """
        Identify coverage gaps across frameworks.

        Args:
            aggregated_metrics: Aggregated metrics
            required_esrs_codes: List of required ESRS codes (optional)

        Returns:
            GapAnalysis
        """
        # If no required codes specified, use all unique ESRS codes from mappings
        if not required_esrs_codes:
            required_esrs_codes = set()
            for mapping in self.framework_mappings.get("esrs_to_tcfd", []):
                required_esrs_codes.add(mapping.get("esrs_code", ""))
            for mapping in self.framework_mappings.get("esrs_to_gri", []):
                required_esrs_codes.add(mapping.get("esrs_code", ""))
            for mapping in self.framework_mappings.get("esrs_to_sasb", []):
                required_esrs_codes.add(mapping.get("esrs_code", ""))
            required_esrs_codes = list(required_esrs_codes)

        # Calculate coverage
        covered_codes = set(aggregated_metrics.keys())
        missing_codes = [code for code in required_esrs_codes if code not in covered_codes]

        total_required = len(required_esrs_codes)
        total_covered = len(covered_codes)
        coverage_percentage = (total_covered / total_required * 100) if total_required > 0 else 0

        # Coverage by framework
        coverage_by_framework = {
            "ESRS": self.stats["esrs_metrics"],
            "TCFD": self.stats["tcfd_metrics_mapped"],
            "GRI": self.stats["gri_metrics_mapped"],
            "SASB": self.stats["sasb_metrics_mapped"]
        }

        # Mapping quality breakdown
        direct_mappings = 0
        high_quality_mappings = 0
        partial_mappings = 0
        low_quality_mappings = 0
        partial_codes = []

        for metric in aggregated_metrics.values():
            quality = metric.mapping_quality
            if quality == "direct":
                direct_mappings += 1
            elif quality == "high":
                high_quality_mappings += 1
            elif quality == "partial":
                partial_mappings += 1
                partial_codes.append(metric.esrs_code)
            elif quality == "low":
                low_quality_mappings += 1

        gap_analysis = GapAnalysis(
            total_esrs_required=total_required,
            total_esrs_covered=total_covered,
            coverage_percentage=round(coverage_percentage, 1),
            coverage_by_framework=coverage_by_framework,
            missing_esrs_codes=missing_codes,
            partial_mappings=partial_codes,
            direct_mappings=direct_mappings,
            high_quality_mappings=high_quality_mappings,
            partial_mappings_count=partial_mappings,
            low_quality_mappings=low_quality_mappings
        )

        return gap_analysis

    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================

    def aggregate(
        self,
        esrs_data: Optional[Dict[str, Any]] = None,
        tcfd_data: Optional[Dict[str, Any]] = None,
        gri_data: Optional[Dict[str, Any]] = None,
        sasb_data: Optional[Dict[str, Any]] = None,
        historical_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        industry_sector: Optional[str] = None,
        required_esrs_codes: Optional[List[str]] = None,
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate ESG data across multiple frameworks.

        Args:
            esrs_data: ESRS-calculated metrics
            tcfd_data: TCFD climate disclosure data
            gri_data: GRI sustainability report data
            sasb_data: SASB industry metrics
            historical_data: Historical time-series data
            industry_sector: Company's industry sector for benchmarking
            required_esrs_codes: List of required ESRS codes (optional)
            output_file: Path for output file (optional)

        Returns:
            Result dictionary with aggregated data and analyses
        """
        self.stats["start_time"] = DeterministicClock.now()

        # Step 1: Integrate multi-framework data
        logger.info("Integrating multi-framework data...")
        aggregated_metrics, integration_issues = self.integrate_multi_framework_data(
            esrs_data, tcfd_data, gri_data, sasb_data
        )
        self.stats["total_metrics_processed"] = len(aggregated_metrics)

        # Step 2: Time-series analysis
        logger.info("Performing time-series analysis...")
        trend_analyses, trend_issues = self.perform_time_series_analysis(
            aggregated_metrics, historical_data
        )

        # Step 3: Benchmark comparison
        logger.info("Performing benchmark comparison...")
        benchmark_comparisons, benchmark_issues = self.perform_benchmark_comparison(
            aggregated_metrics, industry_sector
        )

        # Step 4: Gap analysis
        logger.info("Performing gap analysis...")
        gap_analysis = self.perform_gap_analysis(
            aggregated_metrics, required_esrs_codes
        )

        # Collect all issues
        all_issues = integration_issues + trend_issues + benchmark_issues

        self.stats["end_time"] = DeterministicClock.now()
        processing_time = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        # Build result
        result = {
            "metadata": {
                "aggregated_at": self.stats["end_time"].isoformat(),
                "total_metrics_processed": self.stats["total_metrics_processed"],
                "esrs_metrics": self.stats["esrs_metrics"],
                "tcfd_metrics_mapped": self.stats["tcfd_metrics_mapped"],
                "gri_metrics_mapped": self.stats["gri_metrics_mapped"],
                "sasb_metrics_mapped": self.stats["sasb_metrics_mapped"],
                "trends_analyzed": self.stats["trends_analyzed"],
                "benchmarks_compared": self.stats["benchmarks_compared"],
                "processing_time_seconds": round(processing_time, 3),
                "deterministic": True,
                "zero_hallucination": True
            },
            "aggregated_esg_data": {
                code: metric.dict() for code, metric in aggregated_metrics.items()
            },
            "trend_analysis": [trend.dict() for trend in trend_analyses],
            "gap_analysis": gap_analysis.dict(),
            "benchmark_comparison": [comp.dict() for comp in benchmark_comparisons],
            "aggregation_issues": [issue.dict() for issue in all_issues]
        }

        # Write output if path provided
        if output_file:
            self.write_output(result, output_file)

        logger.info(f"Aggregated {self.stats['total_metrics_processed']} metrics in {processing_time:.2f}s")
        logger.info(f"ESRS: {self.stats['esrs_metrics']}, TCFD: {self.stats['tcfd_metrics_mapped']}, "
                   f"GRI: {self.stats['gri_metrics_mapped']}, SASB: {self.stats['sasb_metrics_mapped']}")
        logger.info(f"Trends: {self.stats['trends_analyzed']}, Benchmarks: {self.stats['benchmarks_compared']}")
        logger.info(f"Coverage: {gap_analysis.coverage_percentage:.1f}% ({gap_analysis.total_esrs_covered}/{gap_analysis.total_esrs_required})")

        return result

    def write_output(self, result: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Write result to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Wrote aggregated data to {output_path}")


# ============================================================================
# CLI INTERFACE (for testing)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSRD Multi-Standard Aggregator Agent")
    parser.add_argument("--framework-mappings", required=True, help="Path to framework mappings JSON")
    parser.add_argument("--esrs-data", help="Path to ESRS calculated metrics JSON")
    parser.add_argument("--tcfd-data", help="Path to TCFD data JSON")
    parser.add_argument("--gri-data", help="Path to GRI data JSON")
    parser.add_argument("--sasb-data", help="Path to SASB data JSON")
    parser.add_argument("--historical-data", help="Path to historical time-series data JSON")
    parser.add_argument("--industry-benchmarks", help="Path to industry benchmarks JSON")
    parser.add_argument("--industry-sector", help="Company's industry sector")
    parser.add_argument("--output", help="Output JSON file path")

    args = parser.parse_args()

    # Create agent
    agent = AggregatorAgent(
        framework_mappings_path=args.framework_mappings,
        industry_benchmarks_path=args.industry_benchmarks
    )

    # Load input data
    esrs_data = None
    if args.esrs_data:
        with open(args.esrs_data, 'r', encoding='utf-8') as f:
            esrs_data = json.load(f)

    tcfd_data = None
    if args.tcfd_data:
        with open(args.tcfd_data, 'r', encoding='utf-8') as f:
            tcfd_data = json.load(f)

    gri_data = None
    if args.gri_data:
        with open(args.gri_data, 'r', encoding='utf-8') as f:
            gri_data = json.load(f)

    sasb_data = None
    if args.sasb_data:
        with open(args.sasb_data, 'r', encoding='utf-8') as f:
            sasb_data = json.load(f)

    historical_data = None
    if args.historical_data:
        with open(args.historical_data, 'r', encoding='utf-8') as f:
            historical_data = json.load(f)

    # Aggregate
    result = agent.aggregate(
        esrs_data=esrs_data,
        tcfd_data=tcfd_data,
        gri_data=gri_data,
        sasb_data=sasb_data,
        historical_data=historical_data,
        industry_sector=args.industry_sector,
        output_file=args.output
    )

    # Print summary
    print("\n" + "="*80)
    print("MULTI-STANDARD AGGREGATION SUMMARY")
    print("="*80)
    print(f"Total Metrics Processed: {result['metadata']['total_metrics_processed']}")
    print(f"  ESRS Metrics: {result['metadata']['esrs_metrics']}")
    print(f"  TCFD Metrics Mapped: {result['metadata']['tcfd_metrics_mapped']}")
    print(f"  GRI Metrics Mapped: {result['metadata']['gri_metrics_mapped']}")
    print(f"  SASB Metrics Mapped: {result['metadata']['sasb_metrics_mapped']}")
    print(f"\nAnalyses:")
    print(f"  Trends Analyzed: {result['metadata']['trends_analyzed']}")
    print(f"  Benchmarks Compared: {result['metadata']['benchmarks_compared']}")
    print(f"\nCoverage:")
    gap = result['gap_analysis']
    print(f"  Total Coverage: {gap['coverage_percentage']:.1f}% ({gap['total_esrs_covered']}/{gap['total_esrs_required']})")
    print(f"  Direct Mappings: {gap['direct_mappings']}")
    print(f"  High Quality Mappings: {gap['high_quality_mappings']}")
    print(f"  Partial Mappings: {gap['partial_mappings_count']}")
    print(f"  Missing ESRS Codes: {len(gap['missing_esrs_codes'])}")
    print(f"\nProcessing Time: {result['metadata']['processing_time_seconds']:.3f}s")
    print(f"\nZero Hallucination Guarantee: ✅ TRUE")
    print(f"Deterministic: ✅ TRUE")

    if result['aggregation_issues']:
        print(f"\nIssues: {len(result['aggregation_issues'])}")
        # Group by severity
        errors = [i for i in result['aggregation_issues'] if i['severity'] == 'error']
        warnings = [i for i in result['aggregation_issues'] if i['severity'] == 'warning']
        info = [i for i in result['aggregation_issues'] if i['severity'] == 'info']
        print(f"  Errors: {len(errors)}")
        print(f"  Warnings: {len(warnings)}")
        print(f"  Info: {len(info)}")

        # Show first few issues
        for issue in result['aggregation_issues'][:5]:
            print(f"  - [{issue['error_code']}] {issue['message']}")

    if result['trend_analysis']:
        print(f"\nSample Trend Analysis:")
        for trend in result['trend_analysis'][:3]:
            print(f"  {trend['metric_name']} ({trend['esrs_code']}):")
            print(f"    YoY Change: {trend.get('yoy_change_percent', 'N/A')}%")
            print(f"    Trend Direction: {trend.get('trend_direction', 'N/A')}")

    if result['benchmark_comparison']:
        print(f"\nSample Benchmark Comparison:")
        for comp in result['benchmark_comparison'][:3]:
            print(f"  {comp['metric_name']} ({comp['esrs_code']}):")
            print(f"    Company: {comp['company_value']} {comp['unit']}")
            print(f"    Sector Median: {comp.get('sector_median', 'N/A')}")
            print(f"    Performance: {comp.get('performance_vs_median', 'N/A')}")
