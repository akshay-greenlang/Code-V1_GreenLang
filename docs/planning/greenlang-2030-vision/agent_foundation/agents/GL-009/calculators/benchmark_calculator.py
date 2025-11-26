"""Industry Benchmark Calculator.

This module compares thermal efficiency against industry benchmarks,
providing percentile rankings, gap analysis, and improvement potential.

Features:
    - Best-in-class targets by process type
    - Percentile ranking against industry data
    - Gap analysis (current vs target)
    - Improvement potential quantification
    - Sector-specific benchmarks

Data Sources:
    - US DOE Industrial Assessment Center
    - EPA ENERGY STAR benchmarks
    - EU BREF documents
    - Industry association data

Author: GL-009 THERMALIQ Agent
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import hashlib
import json
from datetime import datetime
import math


class ProcessType(Enum):
    """Types of thermal processes for benchmarking."""
    STEAM_BOILER_NATURAL_GAS = "steam_boiler_natural_gas"
    STEAM_BOILER_OIL = "steam_boiler_oil"
    STEAM_BOILER_COAL = "steam_boiler_coal"
    STEAM_BOILER_BIOMASS = "steam_boiler_biomass"
    HOT_WATER_BOILER = "hot_water_boiler"
    THERMAL_FLUID_HEATER = "thermal_fluid_heater"
    FURNACE_PROCESS_HEAT = "furnace_process_heat"
    DRYER_DIRECT = "dryer_direct"
    DRYER_INDIRECT = "dryer_indirect"
    OVEN_BATCH = "oven_batch"
    OVEN_CONTINUOUS = "oven_continuous"
    KILN_ROTARY = "kiln_rotary"
    KILN_TUNNEL = "kiln_tunnel"
    HEAT_EXCHANGER = "heat_exchanger"
    CHP_SYSTEM = "chp_system"
    ABSORPTION_CHILLER = "absorption_chiller"
    DISTILLATION_COLUMN = "distillation_column"
    EVAPORATOR = "evaporator"
    OTHER = "other"


class BenchmarkSource(Enum):
    """Source of benchmark data."""
    DOE_IAC = "doe_iac"
    EPA_ENERGY_STAR = "epa_energy_star"
    EU_BREF = "eu_bref"
    ASHRAE = "ashrae"
    ASME = "asme"
    INDUSTRY_SURVEY = "industry_survey"
    MANUFACTURER = "manufacturer"
    CUSTOM = "custom"


@dataclass(frozen=True)
class IndustryBenchmark:
    """Industry benchmark data for a process type.

    Attributes:
        process_type: Type of thermal process
        best_in_class_percent: Top 10% efficiency (%)
        first_quartile_percent: 75th percentile efficiency
        median_percent: 50th percentile efficiency
        third_quartile_percent: 25th percentile efficiency
        minimum_acceptable_percent: Regulatory minimum (if any)
        source: Source of benchmark data
        vintage_year: Year of benchmark data
        notes: Additional notes
    """
    process_type: ProcessType
    best_in_class_percent: float
    first_quartile_percent: float
    median_percent: float
    third_quartile_percent: float
    minimum_acceptable_percent: float
    source: BenchmarkSource
    vintage_year: int
    notes: Optional[str] = None


@dataclass
class PercentileRanking:
    """Percentile ranking result.

    Attributes:
        percentile: Percentile rank (0-100)
        category: Performance category description
        better_than_percent: Percentage of peers performing worse
        distance_to_median_percent: Distance from median (+ is better)
        distance_to_best_percent: Distance from best-in-class
    """
    percentile: float
    category: str
    better_than_percent: float
    distance_to_median_percent: float
    distance_to_best_percent: float


@dataclass
class GapAnalysis:
    """Gap analysis between current and target performance.

    Attributes:
        current_efficiency_percent: Current efficiency
        target_efficiency_percent: Target efficiency
        gap_percent: Efficiency gap (target - current)
        gap_relative_percent: Relative gap (gap / current * 100)
        target_type: Description of target (e.g., "Best-in-class")
        achievability: Qualitative achievability assessment
        estimated_payback_years: Rough payback estimate
    """
    current_efficiency_percent: float
    target_efficiency_percent: float
    gap_percent: float
    gap_relative_percent: float
    target_type: str
    achievability: str
    estimated_payback_years: Optional[float] = None


@dataclass
class CalculationStep:
    """Records a single calculation step for audit trail."""
    step_number: int
    description: str
    operation: str
    inputs: Dict[str, float]
    output_value: float
    output_name: str
    formula: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark comparison result.

    Attributes:
        process_type: Type of thermal process
        current_efficiency_percent: Current efficiency
        benchmark: Industry benchmark used
        percentile_ranking: Ranking among peers
        gap_to_median: Gap to median performance
        gap_to_best: Gap to best-in-class
        improvement_potential_kw: Potential energy savings (kW)
        improvement_potential_percent: Savings as % of input
        recommendations: Prioritized recommendations
        calculation_steps: Audit trail
        provenance_hash: SHA-256 hash
        calculation_timestamp: When calculated
    """
    process_type: ProcessType
    current_efficiency_percent: float
    benchmark: IndustryBenchmark
    percentile_ranking: PercentileRanking
    gap_to_median: GapAnalysis
    gap_to_best: GapAnalysis
    improvement_potential_kw: float
    improvement_potential_percent: float
    recommendations: List[str]
    calculation_steps: List[CalculationStep]
    provenance_hash: str
    calculation_timestamp: str
    calculator_version: str = "1.0.0"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "process_type": self.process_type.value,
            "current_efficiency_percent": self.current_efficiency_percent,
            "benchmark": {
                "best_in_class": self.benchmark.best_in_class_percent,
                "median": self.benchmark.median_percent,
                "source": self.benchmark.source.value,
                "year": self.benchmark.vintage_year
            },
            "percentile_ranking": {
                "percentile": self.percentile_ranking.percentile,
                "category": self.percentile_ranking.category,
                "better_than_percent": self.percentile_ranking.better_than_percent
            },
            "gap_to_best": {
                "gap_percent": self.gap_to_best.gap_percent,
                "achievability": self.gap_to_best.achievability
            },
            "improvement_potential_kw": self.improvement_potential_kw,
            "improvement_potential_percent": self.improvement_potential_percent,
            "recommendations": self.recommendations,
            "provenance_hash": self.provenance_hash
        }


class BenchmarkCalculator:
    """Industry Benchmark Comparison Calculator.

    Compares equipment efficiency against industry benchmarks
    to identify improvement opportunities.

    Example:
        >>> calculator = BenchmarkCalculator()
        >>> result = calculator.compare_to_benchmark(
        ...     process_type=ProcessType.STEAM_BOILER_NATURAL_GAS,
        ...     current_efficiency_percent=82.0,
        ...     energy_input_kw=5000
        ... )
        >>> print(f"Percentile: {result.percentile_ranking.percentile}")
    """

    VERSION: str = "1.0.0"
    PRECISION: int = 2

    # Industry benchmark database
    BENCHMARKS: Dict[ProcessType, IndustryBenchmark] = {
        ProcessType.STEAM_BOILER_NATURAL_GAS: IndustryBenchmark(
            process_type=ProcessType.STEAM_BOILER_NATURAL_GAS,
            best_in_class_percent=92.0,
            first_quartile_percent=88.0,
            median_percent=84.0,
            third_quartile_percent=78.0,
            minimum_acceptable_percent=75.0,
            source=BenchmarkSource.DOE_IAC,
            vintage_year=2023,
            notes="Based on >1000 DOE assessments"
        ),
        ProcessType.STEAM_BOILER_OIL: IndustryBenchmark(
            process_type=ProcessType.STEAM_BOILER_OIL,
            best_in_class_percent=90.0,
            first_quartile_percent=86.0,
            median_percent=82.0,
            third_quartile_percent=76.0,
            minimum_acceptable_percent=72.0,
            source=BenchmarkSource.DOE_IAC,
            vintage_year=2023
        ),
        ProcessType.STEAM_BOILER_COAL: IndustryBenchmark(
            process_type=ProcessType.STEAM_BOILER_COAL,
            best_in_class_percent=88.0,
            first_quartile_percent=84.0,
            median_percent=80.0,
            third_quartile_percent=74.0,
            minimum_acceptable_percent=70.0,
            source=BenchmarkSource.DOE_IAC,
            vintage_year=2023
        ),
        ProcessType.STEAM_BOILER_BIOMASS: IndustryBenchmark(
            process_type=ProcessType.STEAM_BOILER_BIOMASS,
            best_in_class_percent=85.0,
            first_quartile_percent=80.0,
            median_percent=75.0,
            third_quartile_percent=68.0,
            minimum_acceptable_percent=60.0,
            source=BenchmarkSource.EU_BREF,
            vintage_year=2022
        ),
        ProcessType.HOT_WATER_BOILER: IndustryBenchmark(
            process_type=ProcessType.HOT_WATER_BOILER,
            best_in_class_percent=95.0,
            first_quartile_percent=92.0,
            median_percent=88.0,
            third_quartile_percent=82.0,
            minimum_acceptable_percent=78.0,
            source=BenchmarkSource.ASHRAE,
            vintage_year=2023
        ),
        ProcessType.THERMAL_FLUID_HEATER: IndustryBenchmark(
            process_type=ProcessType.THERMAL_FLUID_HEATER,
            best_in_class_percent=88.0,
            first_quartile_percent=84.0,
            median_percent=80.0,
            third_quartile_percent=74.0,
            minimum_acceptable_percent=70.0,
            source=BenchmarkSource.INDUSTRY_SURVEY,
            vintage_year=2022
        ),
        ProcessType.FURNACE_PROCESS_HEAT: IndustryBenchmark(
            process_type=ProcessType.FURNACE_PROCESS_HEAT,
            best_in_class_percent=75.0,
            first_quartile_percent=68.0,
            median_percent=60.0,
            third_quartile_percent=50.0,
            minimum_acceptable_percent=40.0,
            source=BenchmarkSource.DOE_IAC,
            vintage_year=2023
        ),
        ProcessType.DRYER_DIRECT: IndustryBenchmark(
            process_type=ProcessType.DRYER_DIRECT,
            best_in_class_percent=70.0,
            first_quartile_percent=62.0,
            median_percent=55.0,
            third_quartile_percent=45.0,
            minimum_acceptable_percent=35.0,
            source=BenchmarkSource.DOE_IAC,
            vintage_year=2023
        ),
        ProcessType.DRYER_INDIRECT: IndustryBenchmark(
            process_type=ProcessType.DRYER_INDIRECT,
            best_in_class_percent=65.0,
            first_quartile_percent=58.0,
            median_percent=50.0,
            third_quartile_percent=42.0,
            minimum_acceptable_percent=32.0,
            source=BenchmarkSource.DOE_IAC,
            vintage_year=2023
        ),
        ProcessType.CHP_SYSTEM: IndustryBenchmark(
            process_type=ProcessType.CHP_SYSTEM,
            best_in_class_percent=85.0,
            first_quartile_percent=78.0,
            median_percent=72.0,
            third_quartile_percent=65.0,
            minimum_acceptable_percent=55.0,
            source=BenchmarkSource.EPA_ENERGY_STAR,
            vintage_year=2023,
            notes="Combined heat and power - total efficiency"
        ),
        ProcessType.HEAT_EXCHANGER: IndustryBenchmark(
            process_type=ProcessType.HEAT_EXCHANGER,
            best_in_class_percent=98.0,
            first_quartile_percent=95.0,
            median_percent=90.0,
            third_quartile_percent=85.0,
            minimum_acceptable_percent=75.0,
            source=BenchmarkSource.ASHRAE,
            vintage_year=2023,
            notes="Heat exchanger effectiveness"
        ),
    }

    def __init__(self, precision: int = 2) -> None:
        """Initialize the Benchmark Calculator.

        Args:
            precision: Decimal places for rounding
        """
        self.precision = precision
        self._calculation_steps: List[CalculationStep] = []
        self._step_counter: int = 0
        self._warnings: List[str] = []

    def compare_to_benchmark(
        self,
        process_type: ProcessType,
        current_efficiency_percent: float,
        energy_input_kw: float,
        custom_benchmark: Optional[IndustryBenchmark] = None
    ) -> BenchmarkResult:
        """Compare current efficiency to industry benchmark.

        Args:
            process_type: Type of thermal process
            current_efficiency_percent: Current operating efficiency
            energy_input_kw: Energy input rate for savings calc
            custom_benchmark: Optional custom benchmark to use

        Returns:
            BenchmarkResult with complete analysis
        """
        self._reset_calculation_state()

        # Get benchmark
        if custom_benchmark:
            benchmark = custom_benchmark
        elif process_type in self.BENCHMARKS:
            benchmark = self.BENCHMARKS[process_type]
        else:
            raise ValueError(f"No benchmark available for {process_type}")

        # Calculate percentile ranking
        percentile_ranking = self._calculate_percentile(
            current_efficiency_percent, benchmark
        )

        # Calculate gap to median
        gap_to_median = self._calculate_gap(
            current_efficiency_percent,
            benchmark.median_percent,
            "Industry Median"
        )

        # Calculate gap to best-in-class
        gap_to_best = self._calculate_gap(
            current_efficiency_percent,
            benchmark.best_in_class_percent,
            "Best-in-Class"
        )

        # Calculate improvement potential
        improvement_pct = benchmark.best_in_class_percent - current_efficiency_percent
        # Energy saved = input * (1/current_eff - 1/target_eff) * target_eff
        # Simplified: savings = input * improvement / current
        improvement_kw = energy_input_kw * (improvement_pct / 100)

        self._add_calculation_step(
            description="Calculate improvement potential",
            operation="improvement_calc",
            inputs={
                "current_efficiency": current_efficiency_percent,
                "best_in_class": benchmark.best_in_class_percent,
                "energy_input_kw": energy_input_kw
            },
            output_value=improvement_kw,
            output_name="improvement_potential_kw",
            formula="Savings = Input x (Target - Current) / 100"
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_efficiency_percent, benchmark, percentile_ranking
        )

        # Generate provenance
        provenance = self._generate_provenance_hash(
            process_type, current_efficiency_percent, benchmark
        )
        timestamp = datetime.utcnow().isoformat() + "Z"

        return BenchmarkResult(
            process_type=process_type,
            current_efficiency_percent=current_efficiency_percent,
            benchmark=benchmark,
            percentile_ranking=percentile_ranking,
            gap_to_median=gap_to_median,
            gap_to_best=gap_to_best,
            improvement_potential_kw=self._round_value(improvement_kw),
            improvement_potential_percent=self._round_value(improvement_pct),
            recommendations=recommendations,
            calculation_steps=self._calculation_steps.copy(),
            provenance_hash=provenance,
            calculation_timestamp=timestamp,
            warnings=self._warnings.copy()
        )

    def get_benchmark(self, process_type: ProcessType) -> IndustryBenchmark:
        """Get industry benchmark for a process type.

        Args:
            process_type: Type of thermal process

        Returns:
            IndustryBenchmark for the process
        """
        if process_type not in self.BENCHMARKS:
            raise ValueError(f"No benchmark available for {process_type}")
        return self.BENCHMARKS[process_type]

    def list_available_benchmarks(self) -> List[ProcessType]:
        """List all available benchmark process types.

        Returns:
            List of ProcessType enums with benchmarks
        """
        return list(self.BENCHMARKS.keys())

    def _calculate_percentile(
        self,
        efficiency: float,
        benchmark: IndustryBenchmark
    ) -> PercentileRanking:
        """Calculate percentile ranking from benchmark data.

        Uses linear interpolation between quartiles.
        """
        # Determine percentile using quartile interpolation
        if efficiency >= benchmark.best_in_class_percent:
            percentile = 95.0  # Top 5%
            category = "Best-in-Class"
        elif efficiency >= benchmark.first_quartile_percent:
            # Interpolate between 75th and 95th
            range_size = benchmark.best_in_class_percent - benchmark.first_quartile_percent
            position = efficiency - benchmark.first_quartile_percent
            percentile = 75 + (position / range_size * 20) if range_size > 0 else 75
            category = "First Quartile"
        elif efficiency >= benchmark.median_percent:
            # Interpolate between 50th and 75th
            range_size = benchmark.first_quartile_percent - benchmark.median_percent
            position = efficiency - benchmark.median_percent
            percentile = 50 + (position / range_size * 25) if range_size > 0 else 50
            category = "Above Median"
        elif efficiency >= benchmark.third_quartile_percent:
            # Interpolate between 25th and 50th
            range_size = benchmark.median_percent - benchmark.third_quartile_percent
            position = efficiency - benchmark.third_quartile_percent
            percentile = 25 + (position / range_size * 25) if range_size > 0 else 25
            category = "Below Median"
        elif efficiency >= benchmark.minimum_acceptable_percent:
            # Interpolate between min and 25th
            range_size = benchmark.third_quartile_percent - benchmark.minimum_acceptable_percent
            position = efficiency - benchmark.minimum_acceptable_percent
            percentile = 5 + (position / range_size * 20) if range_size > 0 else 5
            category = "Third Quartile"
        else:
            percentile = 5.0  # Bottom 5%
            category = "Below Minimum"
            self._warnings.append(
                f"Efficiency {efficiency}% is below minimum acceptable"
            )

        distance_to_median = efficiency - benchmark.median_percent
        distance_to_best = efficiency - benchmark.best_in_class_percent

        self._add_calculation_step(
            description="Calculate percentile ranking",
            operation="percentile_interpolation",
            inputs={
                "current_efficiency": efficiency,
                "median": benchmark.median_percent,
                "best_in_class": benchmark.best_in_class_percent
            },
            output_value=percentile,
            output_name="percentile",
            formula="Interpolation between quartiles"
        )

        return PercentileRanking(
            percentile=self._round_value(percentile),
            category=category,
            better_than_percent=self._round_value(percentile),
            distance_to_median_percent=self._round_value(distance_to_median),
            distance_to_best_percent=self._round_value(distance_to_best)
        )

    def _calculate_gap(
        self,
        current: float,
        target: float,
        target_type: str
    ) -> GapAnalysis:
        """Calculate gap between current and target efficiency."""
        gap = target - current
        gap_relative = (gap / current * 100) if current > 0 else 0

        # Assess achievability
        if gap <= 0:
            achievability = "Already achieved"
        elif gap <= 2:
            achievability = "Easily achievable with operational improvements"
        elif gap <= 5:
            achievability = "Achievable with moderate investment"
        elif gap <= 10:
            achievability = "Requires significant investment"
        else:
            achievability = "Major upgrade or replacement needed"

        # Rough payback estimate (years)
        # Assumes $0.05/kWh energy cost, $500/kW improvement cost
        if gap > 0:
            payback = 2.0 + (gap / 5)  # Very rough estimate
        else:
            payback = None

        self._add_calculation_step(
            description=f"Calculate gap to {target_type}",
            operation="gap_analysis",
            inputs={"current": current, "target": target},
            output_value=gap,
            output_name="gap_percent",
            formula="Gap = Target - Current"
        )

        return GapAnalysis(
            current_efficiency_percent=self._round_value(current),
            target_efficiency_percent=self._round_value(target),
            gap_percent=self._round_value(gap),
            gap_relative_percent=self._round_value(gap_relative),
            target_type=target_type,
            achievability=achievability,
            estimated_payback_years=self._round_value(payback) if payback else None
        )

    def _generate_recommendations(
        self,
        current: float,
        benchmark: IndustryBenchmark,
        ranking: PercentileRanking
    ) -> List[str]:
        """Generate prioritized recommendations based on gap analysis."""
        recommendations = []

        gap_to_best = benchmark.best_in_class_percent - current
        gap_to_median = benchmark.median_percent - current

        if ranking.percentile < 25:
            # Bottom quartile - urgent action needed
            recommendations.append(
                "PRIORITY: Conduct comprehensive energy audit to identify major losses"
            )
            recommendations.append(
                "Check for combustion efficiency issues (excess air, incomplete combustion)"
            )
            recommendations.append(
                "Inspect insulation for damage or degradation"
            )
            recommendations.append(
                "Verify instrumentation accuracy for efficiency calculations"
            )

        elif ranking.percentile < 50:
            # Third quartile
            recommendations.append(
                "Optimize combustion with O2 trim controls"
            )
            recommendations.append(
                "Implement economizer for flue gas heat recovery"
            )
            recommendations.append(
                "Review blowdown practices and consider heat recovery"
            )
            recommendations.append(
                "Consider variable frequency drives for fans/pumps"
            )

        elif ranking.percentile < 75:
            # Second quartile
            recommendations.append(
                "Install or upgrade economizer for enhanced heat recovery"
            )
            recommendations.append(
                "Implement condensing technology if feasible"
            )
            recommendations.append(
                "Optimize steam system (trap maintenance, insulation)"
            )
            recommendations.append(
                "Consider air preheater for combustion air"
            )

        else:
            # First quartile or better
            recommendations.append(
                "Maintain current best practices and monitoring"
            )
            recommendations.append(
                "Consider advanced controls (model predictive control)"
            )
            if gap_to_best > 2:
                recommendations.append(
                    "Evaluate condensing heat exchangers for additional recovery"
                )
            recommendations.append(
                "Benchmark against peers to identify remaining opportunities"
            )

        # Add savings potential
        if gap_to_best > 0:
            recommendations.append(
                f"Improvement potential: {gap_to_best:.1f} percentage points to best-in-class"
            )

        return recommendations

    def _reset_calculation_state(self) -> None:
        """Reset calculation state."""
        self._calculation_steps = []
        self._step_counter = 0
        self._warnings = []

    def _add_calculation_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, float],
        output_value: float,
        output_name: str,
        formula: Optional[str] = None
    ) -> None:
        """Record a calculation step."""
        self._step_counter += 1
        step = CalculationStep(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula
        )
        self._calculation_steps.append(step)

    def _generate_provenance_hash(
        self,
        process_type: ProcessType,
        efficiency: float,
        benchmark: IndustryBenchmark
    ) -> str:
        """Generate SHA-256 provenance hash."""
        data = {
            "calculator": "BenchmarkCalculator",
            "version": self.VERSION,
            "process_type": process_type.value,
            "current_efficiency": efficiency,
            "benchmark_source": benchmark.source.value,
            "benchmark_year": benchmark.vintage_year
        }
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _round_value(self, value: float) -> float:
        """Round value to precision."""
        if value is None:
            return 0.0
        decimal_value = Decimal(str(value))
        quantize_str = '0.' + '0' * self.precision
        rounded = decimal_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )
        return float(rounded)
