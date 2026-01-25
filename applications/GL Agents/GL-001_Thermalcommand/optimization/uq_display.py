"""
UQ Display - Visualization data generation for uncertainty quantification

This module generates display data for uncertainty visualization in
dashboards and reports. All calculations are deterministic with
SHA-256 provenance tracking.

Key Components:
    - UQDisplayEngine: Main engine for display data generation
    - FanChartGenerator: Generate fan chart data
    - RiskAssessmentEngine: Constraint binding probability assessment

Reference Standards:
    - ISO 31000 (Risk Management)
    - ASME PTC 19.1 (Test Uncertainty)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
import math

from .uq_schemas import (
    ProvenanceRecord,
    PredictionInterval,
    QuantileSet,
    QuantileValue,
    Scenario,
    ScenarioSet,
    UncertaintyBand,
    FanChartData,
    RiskAssessment,
    ScenarioComparison,
    RobustSolution,
)
from .uncertainty_models import UncertaintyModelEngine, UncertaintySource


class UQDisplayEngine:
    """
    Main engine for UQ visualization data - ZERO HALLUCINATION.

    Generates display-ready data for:
    - Expected values with P10/P90 ranges
    - Fan charts for uncertainty visualization
    - Risk assessments for constraint monitoring
    - Scenario comparisons for decision support

    All calculations are deterministic with provenance tracking.
    """

    def __init__(self, uncertainty_engine: Optional[UncertaintyModelEngine] = None):
        """Initialize UQ display engine."""
        self.uncertainty_engine = uncertainty_engine or UncertaintyModelEngine()
        self._fan_chart_generator = FanChartGenerator(self.uncertainty_engine)
        self._risk_engine = RiskAssessmentEngine()

    def generate_summary_display(
        self,
        prediction: PredictionInterval,
        include_risk: bool = True
    ) -> Dict[str, Any]:
        """
        Generate summary display data - DETERMINISTIC.

        Args:
            prediction: Prediction interval
            include_risk: Whether to include risk assessment

        Returns:
            Dictionary with display-ready data
        """
        start_time = time.time()

        # Basic summary
        summary = {
            "variable_name": prediction.variable_name,
            "unit": prediction.unit,
            "expected_value": str(prediction.point_estimate),
            "p10": str(prediction.lower_bound),
            "p90": str(prediction.upper_bound),
            "confidence_level": str(prediction.confidence_level),
            "interval_width": str(prediction.interval_width),
            "relative_width_percent": str(
                (prediction.relative_width * Decimal("100")).quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP
                )
            ),
            "horizon_minutes": prediction.horizon_minutes,
            "timestamp": prediction.timestamp.isoformat(),
            "source_model": prediction.source_model
        }

        # Display formatting
        summary["display"] = {
            "expected": self._format_value(prediction.point_estimate, prediction.unit),
            "range": f"{self._format_value(prediction.lower_bound, prediction.unit)} - {self._format_value(prediction.upper_bound, prediction.unit)}",
            "confidence": f"{int(float(prediction.confidence_level) * 100)}%"
        }

        # Provenance
        computation_time_ms = (time.time() - start_time) * 1000
        summary["provenance"] = {
            "hash": prediction.provenance.combined_hash if prediction.provenance else None,
            "computation_time_ms": computation_time_ms
        }

        return summary

    def generate_multi_quantile_display(
        self,
        quantile_set: QuantileSet,
        highlight_quantiles: List[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Generate multi-quantile display data - DETERMINISTIC.

        Args:
            quantile_set: Set of quantile values
            highlight_quantiles: Quantiles to highlight (default: P10, P50, P90)

        Returns:
            Dictionary with display-ready quantile data
        """
        if highlight_quantiles is None:
            highlight_quantiles = [Decimal("0.10"), Decimal("0.50"), Decimal("0.90")]

        display = {
            "variable_name": quantile_set.variable_name,
            "unit": quantile_set.unit,
            "timestamp": quantile_set.timestamp.isoformat(),
            "quantiles": [],
            "highlighted": {}
        }

        for qv in quantile_set.quantiles:
            quantile_data = {
                "probability": str(qv.probability),
                "percentile": f"P{int(float(qv.probability) * 100)}",
                "value": str(qv.value),
                "formatted": self._format_value(qv.value, quantile_set.unit)
            }
            display["quantiles"].append(quantile_data)

            if qv.probability in highlight_quantiles:
                display["highlighted"][f"p{int(float(qv.probability) * 100)}"] = {
                    "value": str(qv.value),
                    "formatted": self._format_value(qv.value, quantile_set.unit)
                }

        # Add spread metrics
        if quantile_set.p10 and quantile_set.p90:
            spread = quantile_set.p90 - quantile_set.p10
            display["spread"] = {
                "p10_p90_range": str(spread),
                "formatted": self._format_value(spread, quantile_set.unit)
            }

        return display

    def generate_scenario_display(
        self,
        scenario_set: ScenarioSet,
        metric_name: str,
        decisions: Optional[Dict[str, Decimal]] = None
    ) -> Dict[str, Any]:
        """
        Generate scenario comparison display - DETERMINISTIC.

        Args:
            scenario_set: Set of scenarios
            metric_name: Metric to compare across scenarios
            decisions: Optional decision values for evaluation

        Returns:
            Dictionary with scenario comparison display data
        """
        start_time = time.time()

        # Collect metric values from scenarios
        values_by_scenario = {}
        for scenario in scenario_set.scenarios:
            var = scenario.get_variable(metric_name)
            if var:
                values_by_scenario[scenario.name] = var.value

        if not values_by_scenario:
            return {"error": f"Metric {metric_name} not found in scenarios"}

        # Compute statistics
        values = list(values_by_scenario.values())
        probs = [scenario_set.scenarios[i].probability for i, s in enumerate(scenario_set.scenarios) if s.name in values_by_scenario]

        # Weighted statistics
        expected = sum(v * p for v, p in zip(values, probs))
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        # Standard deviation
        variance = sum(p * (v - expected) ** 2 for v, p in zip(values, probs))
        std_dev = variance.sqrt() if variance > 0 else Decimal("0")

        # Get unit from first variable
        unit = ""
        for scenario in scenario_set.scenarios:
            var = scenario.get_variable(metric_name)
            if var:
                unit = var.unit
                break

        computation_time_ms = (time.time() - start_time) * 1000

        display = {
            "metric_name": metric_name,
            "unit": unit,
            "num_scenarios": scenario_set.num_scenarios,
            "statistics": {
                "expected_value": str(expected.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
                "minimum": str(min_val),
                "maximum": str(max_val),
                "range": str(range_val),
                "std_dev": str(std_dev.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
            },
            "display": {
                "expected": self._format_value(expected, unit),
                "range": f"{self._format_value(min_val, unit)} - {self._format_value(max_val, unit)}"
            },
            "scenarios": [
                {
                    "name": name,
                    "value": str(value),
                    "formatted": self._format_value(value, unit),
                    "is_base_case": any(
                        s.is_base_case for s in scenario_set.scenarios if s.name == name
                    ),
                    "is_worst_case": any(
                        s.is_worst_case for s in scenario_set.scenarios if s.name == name
                    )
                }
                for name, value in values_by_scenario.items()
            ],
            "provenance": {
                "hash": scenario_set.provenance.combined_hash if scenario_set.provenance else None,
                "computation_time_ms": computation_time_ms
            }
        }

        return display

    def generate_solution_display(
        self,
        solution: RobustSolution,
        scenario_set: ScenarioSet
    ) -> Dict[str, Any]:
        """
        Generate robust solution display - DETERMINISTIC.

        Args:
            solution: Robust optimization solution
            scenario_set: Associated scenario set

        Returns:
            Dictionary with solution display data
        """
        display = {
            "solution_id": str(solution.solution_id),
            "objective_type": solution.objective_type.value,
            "objective_value": str(solution.objective_value),
            "solver_status": solution.solver_status,
            "solve_time_ms": solution.solve_time_ms,
            "feasibility": {
                "rate": str(solution.feasibility_rate),
                "percent": f"{int(float(solution.feasibility_rate) * 100)}%",
                "is_fully_feasible": solution.feasibility_rate == Decimal("1.0")
            },
            "decisions": [
                {
                    "name": name,
                    "value": str(value),
                    "formatted": self._format_value(value, "")
                }
                for name, value in solution.decision_variables.items()
            ],
            "risk_metrics": {
                "expected_objective": str(solution.expected_objective) if solution.expected_objective else None,
                "worst_case_objective": str(solution.worst_case_objective) if solution.worst_case_objective else None,
                "cvar": str(solution.cvar) if solution.cvar else None
            },
            "constraints": {
                "binding": solution.binding_constraints,
                "num_binding": len(solution.binding_constraints)
            },
            "timestamp": solution.timestamp.isoformat(),
            "provenance": {
                "hash": solution.provenance.combined_hash if solution.provenance else None
            }
        }

        # Add scenario insights
        if solution.expected_objective and solution.worst_case_objective:
            regret = solution.worst_case_objective - solution.expected_objective
            display["scenario_insights"] = {
                "expected_to_worst_gap": str(regret),
                "risk_exposure": "high" if regret > solution.expected_objective * Decimal("0.5") else "moderate" if regret > solution.expected_objective * Decimal("0.2") else "low"
            }

        return display

    def _format_value(self, value: Decimal, unit: str) -> str:
        """Format value with unit for display."""
        # Format based on magnitude
        abs_val = abs(value)

        if abs_val >= 1000000:
            formatted = f"{float(value / 1000000):.2f}M"
        elif abs_val >= 1000:
            formatted = f"{float(value / 1000):.2f}k"
        elif abs_val >= 1:
            formatted = f"{float(value):.2f}"
        elif abs_val >= 0.01:
            formatted = f"{float(value):.3f}"
        else:
            formatted = f"{float(value):.4f}"

        if unit:
            return f"{formatted} {unit}"
        return formatted


class FanChartGenerator:
    """
    Generate fan chart data for uncertainty visualization - ZERO HALLUCINATION.

    Fan charts show prediction uncertainty over time with
    multiple confidence bands (e.g., 50%, 80%, 90%, 95%).
    """

    DEFAULT_CONFIDENCE_LEVELS = [
        Decimal("0.50"),
        Decimal("0.80"),
        Decimal("0.90"),
        Decimal("0.95")
    ]

    def __init__(self, uncertainty_engine: Optional[UncertaintyModelEngine] = None):
        """Initialize fan chart generator."""
        self.uncertainty_engine = uncertainty_engine or UncertaintyModelEngine()

    def generate(
        self,
        point_forecasts: List[Decimal],
        timestamps: List[datetime],
        uncertainty_source: UncertaintySource,
        confidence_levels: Optional[List[Decimal]] = None,
        historical_values: Optional[List[Decimal]] = None,
        historical_timestamps: Optional[List[datetime]] = None
    ) -> FanChartData:
        """
        Generate fan chart data - DETERMINISTIC.

        Args:
            point_forecasts: Central forecast values
            timestamps: Forecast timestamps
            uncertainty_source: Uncertainty source for bands
            confidence_levels: Confidence levels for bands
            historical_values: Optional historical data
            historical_timestamps: Timestamps for historical data

        Returns:
            FanChartData for visualization
        """
        start_time = time.time()

        if confidence_levels is None:
            confidence_levels = self.DEFAULT_CONFIDENCE_LEVELS

        if len(point_forecasts) != len(timestamps):
            raise ValueError("Point forecasts and timestamps must have same length")

        # Generate bands for each confidence level
        bands: Dict[str, List[Tuple[Decimal, Decimal]]] = {}

        for conf_level in confidence_levels:
            band_key = f"{int(float(conf_level) * 100)}%"
            bands[band_key] = []

            for i, (forecast, ts) in enumerate(zip(point_forecasts, timestamps)):
                # Calculate horizon in minutes from first timestamp
                if i == 0:
                    horizon_minutes = 60
                else:
                    delta = ts - timestamps[0]
                    horizon_minutes = max(1, int(delta.total_seconds() / 60))

                # Generate prediction interval
                interval = self.uncertainty_engine.generate_prediction_interval(
                    point_estimate=forecast,
                    uncertainty_source=uncertainty_source,
                    confidence_level=conf_level,
                    horizon_minutes=horizon_minutes
                )

                bands[band_key].append((interval.lower_bound, interval.upper_bound))

        # Create provenance
        computation_time_ms = (time.time() - start_time) * 1000
        provenance = ProvenanceRecord.create(
            calculation_type="fan_chart_generation",
            inputs={
                "num_forecasts": len(point_forecasts),
                "confidence_levels": [str(c) for c in confidence_levels],
                "uncertainty_source": uncertainty_source.name
            },
            outputs={
                "num_bands": len(bands)
            },
            computation_time_ms=computation_time_ms
        )

        return FanChartData(
            variable_name=uncertainty_source.name,
            unit=uncertainty_source.unit,
            timestamps=timestamps,
            central_values=point_forecasts,
            bands=bands,
            historical_values=historical_values,
            historical_timestamps=historical_timestamps,
            provenance=provenance
        )

    def generate_from_scenarios(
        self,
        scenario_set: ScenarioSet,
        variable_name: str,
        timestamps: List[datetime],
        confidence_levels: Optional[List[Decimal]] = None
    ) -> FanChartData:
        """
        Generate fan chart from scenarios - DETERMINISTIC.

        Uses empirical quantiles from scenario values.
        """
        start_time = time.time()

        if confidence_levels is None:
            confidence_levels = self.DEFAULT_CONFIDENCE_LEVELS

        # Extract values for variable across scenarios
        scenario_values = []
        unit = ""

        for scenario in scenario_set.scenarios:
            var = scenario.get_variable(variable_name)
            if var:
                scenario_values.append(var.value)
                unit = var.unit

        if not scenario_values:
            raise ValueError(f"Variable {variable_name} not found in scenarios")

        # Sort values for quantile computation
        sorted_values = sorted(scenario_values)
        n = len(sorted_values)

        # Compute central value (median or expected)
        central_values = []
        if n % 2 == 0:
            median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / Decimal("2")
        else:
            median = sorted_values[n // 2]

        # Replicate for all timestamps (static forecast from scenarios)
        central_values = [median] * len(timestamps)

        # Generate bands
        bands: Dict[str, List[Tuple[Decimal, Decimal]]] = {}

        for conf_level in confidence_levels:
            band_key = f"{int(float(conf_level) * 100)}%"
            alpha = (Decimal("1") - conf_level) / Decimal("2")

            # Compute empirical quantiles
            lower_idx = max(0, int(float(alpha) * n) - 1)
            upper_idx = min(n - 1, int(float(Decimal("1") - alpha) * n))

            lower_bound = sorted_values[lower_idx]
            upper_bound = sorted_values[upper_idx]

            # Replicate for all timestamps
            bands[band_key] = [(lower_bound, upper_bound)] * len(timestamps)

        computation_time_ms = (time.time() - start_time) * 1000
        provenance = ProvenanceRecord.create(
            calculation_type="fan_chart_from_scenarios",
            inputs={
                "scenario_set_id": str(scenario_set.set_id),
                "variable_name": variable_name,
                "num_scenarios": len(scenario_values),
                "confidence_levels": [str(c) for c in confidence_levels]
            },
            outputs={
                "median": str(median),
                "num_bands": len(bands)
            },
            computation_time_ms=computation_time_ms
        )

        return FanChartData(
            variable_name=variable_name,
            unit=unit,
            timestamps=timestamps,
            central_values=central_values,
            bands=bands,
            provenance=provenance
        )

    def to_chart_format(
        self,
        fan_chart: FanChartData,
        format_type: str = "highcharts"
    ) -> Dict[str, Any]:
        """
        Convert fan chart to specific charting library format - DETERMINISTIC.

        Args:
            fan_chart: FanChartData to convert
            format_type: Target format (highcharts, plotly, chartjs)

        Returns:
            Dictionary in target chart format
        """
        if format_type == "highcharts":
            return self._to_highcharts(fan_chart)
        elif format_type == "plotly":
            return self._to_plotly(fan_chart)
        elif format_type == "chartjs":
            return self._to_chartjs(fan_chart)
        else:
            return self._to_generic(fan_chart)

    def _to_highcharts(self, fan_chart: FanChartData) -> Dict[str, Any]:
        """Convert to Highcharts arearange format."""
        series = []

        # Central line
        series.append({
            "name": "Expected",
            "type": "line",
            "data": [
                [int(ts.timestamp() * 1000), float(v)]
                for ts, v in zip(fan_chart.timestamps, fan_chart.central_values)
            ],
            "zIndex": 1
        })

        # Confidence bands (from widest to narrowest for layering)
        band_colors = {
            "95%": "rgba(124, 181, 236, 0.2)",
            "90%": "rgba(124, 181, 236, 0.3)",
            "80%": "rgba(124, 181, 236, 0.4)",
            "50%": "rgba(124, 181, 236, 0.5)"
        }

        for band_name in sorted(fan_chart.bands.keys(), reverse=True):
            band_values = fan_chart.bands[band_name]
            series.append({
                "name": f"{band_name} CI",
                "type": "arearange",
                "data": [
                    [int(ts.timestamp() * 1000), float(lower), float(upper)]
                    for ts, (lower, upper) in zip(fan_chart.timestamps, band_values)
                ],
                "color": band_colors.get(band_name, "rgba(124, 181, 236, 0.3)"),
                "zIndex": 0
            })

        # Historical values if available
        if fan_chart.historical_values and fan_chart.historical_timestamps:
            series.append({
                "name": "Actual",
                "type": "line",
                "data": [
                    [int(ts.timestamp() * 1000), float(v)]
                    for ts, v in zip(fan_chart.historical_timestamps, fan_chart.historical_values)
                ],
                "color": "#333333",
                "dashStyle": "Dot"
            })

        return {
            "chart": {"type": "line"},
            "title": {"text": f"{fan_chart.variable_name} Forecast"},
            "xAxis": {"type": "datetime"},
            "yAxis": {"title": {"text": fan_chart.unit}},
            "series": series,
            "provenance_hash": fan_chart.provenance.combined_hash if fan_chart.provenance else None
        }

    def _to_plotly(self, fan_chart: FanChartData) -> Dict[str, Any]:
        """Convert to Plotly format."""
        traces = []

        timestamps_str = [ts.isoformat() for ts in fan_chart.timestamps]

        # Central line
        traces.append({
            "type": "scatter",
            "name": "Expected",
            "x": timestamps_str,
            "y": [float(v) for v in fan_chart.central_values],
            "mode": "lines",
            "line": {"color": "blue"}
        })

        # Confidence bands
        for band_name, band_values in fan_chart.bands.items():
            lower = [float(l) for l, u in band_values]
            upper = [float(u) for l, u in band_values]

            traces.append({
                "type": "scatter",
                "name": f"{band_name} Upper",
                "x": timestamps_str,
                "y": upper,
                "mode": "lines",
                "line": {"width": 0},
                "showlegend": False
            })
            traces.append({
                "type": "scatter",
                "name": f"{band_name} CI",
                "x": timestamps_str,
                "y": lower,
                "mode": "lines",
                "line": {"width": 0},
                "fill": "tonexty",
                "fillcolor": "rgba(124, 181, 236, 0.3)"
            })

        return {
            "data": traces,
            "layout": {
                "title": f"{fan_chart.variable_name} Forecast",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": fan_chart.unit}
            }
        }

    def _to_chartjs(self, fan_chart: FanChartData) -> Dict[str, Any]:
        """Convert to Chart.js format."""
        labels = [ts.isoformat() for ts in fan_chart.timestamps]

        datasets = [{
            "label": "Expected",
            "data": [float(v) for v in fan_chart.central_values],
            "borderColor": "blue",
            "fill": False
        }]

        return {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": datasets
            },
            "options": {
                "title": {"display": True, "text": f"{fan_chart.variable_name} Forecast"}
            }
        }

    def _to_generic(self, fan_chart: FanChartData) -> Dict[str, Any]:
        """Convert to generic format."""
        return {
            "variable": fan_chart.variable_name,
            "unit": fan_chart.unit,
            "timestamps": [ts.isoformat() for ts in fan_chart.timestamps],
            "central": [str(v) for v in fan_chart.central_values],
            "bands": {
                name: [(str(l), str(u)) for l, u in values]
                for name, values in fan_chart.bands.items()
            }
        }


class RiskAssessmentEngine:
    """
    Assess risk of constraint binding under uncertainty - ZERO HALLUCINATION.

    Evaluates probability that constraints will become binding
    based on current state and uncertainty forecasts.
    """

    def __init__(self):
        """Initialize risk assessment engine."""
        pass

    def assess_constraint_risk(
        self,
        constraint_name: str,
        current_value: Decimal,
        constraint_bound: Decimal,
        bound_type: str,
        uncertainty_std: Decimal,
        horizon_minutes: int = 60
    ) -> RiskAssessment:
        """
        Assess risk of constraint binding - DETERMINISTIC.

        Args:
            constraint_name: Name of constraint
            current_value: Current value of constrained quantity
            constraint_bound: Constraint bound value
            bound_type: Bound type (<=, >=)
            uncertainty_std: Standard deviation of uncertainty
            horizon_minutes: Assessment horizon

        Returns:
            RiskAssessment with binding probability
        """
        start_time = time.time()

        # Calculate headroom
        if bound_type == "<=":
            headroom = constraint_bound - current_value
        else:  # >=
            headroom = current_value - constraint_bound

        # Headroom as percentage
        if constraint_bound != 0:
            headroom_percent = (headroom / abs(constraint_bound)) * Decimal("100")
        else:
            headroom_percent = Decimal("100") if headroom > 0 else Decimal("0")

        # Scale uncertainty with horizon
        horizon_scale = Decimal(str(math.sqrt(horizon_minutes / 60.0)))
        scaled_std = uncertainty_std * horizon_scale

        # Compute probability of binding (assuming normal distribution)
        if scaled_std > 0:
            z_binding = headroom / scaled_std
            prob_binding = self._normal_cdf(-z_binding)
        else:
            prob_binding = Decimal("0") if headroom > 0 else Decimal("1")

        # Probability of violation (more severe)
        violation_threshold = headroom - scaled_std  # 1 std past bound
        if scaled_std > 0:
            z_violation = violation_threshold / scaled_std
            prob_violation = self._normal_cdf(-z_violation)
        else:
            prob_violation = Decimal("0") if violation_threshold > 0 else Decimal("1")

        # Time to binding estimate (if trending toward bound)
        time_to_binding = None
        if headroom > 0 and scaled_std > 0:
            # Estimate based on uncertainty growth
            time_to_binding = int(float(headroom / scaled_std) ** 2 * 60)

        # Determine risk level and recommendation
        if prob_violation > Decimal("0.10"):
            risk_level = "critical"
            recommended_action = f"Immediate action required: {constraint_name} at high violation risk"
        elif prob_binding > Decimal("0.50"):
            risk_level = "high"
            recommended_action = f"Prepare contingency for {constraint_name}"
        elif prob_binding > Decimal("0.20"):
            risk_level = "medium"
            recommended_action = f"Monitor {constraint_name} closely"
        else:
            risk_level = "low"
            recommended_action = None

        computation_time_ms = (time.time() - start_time) * 1000
        provenance = ProvenanceRecord.create(
            calculation_type="constraint_risk_assessment",
            inputs={
                "constraint_name": constraint_name,
                "current_value": str(current_value),
                "constraint_bound": str(constraint_bound),
                "bound_type": bound_type,
                "uncertainty_std": str(uncertainty_std),
                "horizon_minutes": horizon_minutes
            },
            outputs={
                "probability_of_binding": str(prob_binding),
                "probability_of_violation": str(prob_violation),
                "risk_level": risk_level
            },
            computation_time_ms=computation_time_ms
        )

        return RiskAssessment(
            constraint_name=constraint_name,
            current_value=current_value,
            constraint_bound=constraint_bound,
            headroom=headroom,
            headroom_percent=headroom_percent.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP),
            probability_of_binding=prob_binding.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            probability_of_violation=prob_violation.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            time_to_binding_minutes=time_to_binding,
            risk_level=risk_level,
            recommended_action=recommended_action,
            provenance=provenance
        )

    def assess_multiple_constraints(
        self,
        constraints: List[Dict[str, Any]],
        horizon_minutes: int = 60
    ) -> List[RiskAssessment]:
        """
        Assess risk for multiple constraints - DETERMINISTIC.

        Args:
            constraints: List of constraint specifications
            horizon_minutes: Assessment horizon

        Returns:
            List of RiskAssessments sorted by risk level
        """
        assessments = []

        for constraint in constraints:
            assessment = self.assess_constraint_risk(
                constraint_name=constraint["name"],
                current_value=Decimal(str(constraint["current_value"])),
                constraint_bound=Decimal(str(constraint["bound"])),
                bound_type=constraint.get("bound_type", "<="),
                uncertainty_std=Decimal(str(constraint.get("uncertainty_std", "0.1"))),
                horizon_minutes=horizon_minutes
            )
            assessments.append(assessment)

        # Sort by risk level (critical > high > medium > low)
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        assessments.sort(key=lambda a: risk_order.get(a.risk_level, 4))

        return assessments

    def generate_risk_dashboard_data(
        self,
        assessments: List[RiskAssessment]
    ) -> Dict[str, Any]:
        """
        Generate risk dashboard display data - DETERMINISTIC.

        Args:
            assessments: List of risk assessments

        Returns:
            Dictionary with dashboard-ready data
        """
        # Group by risk level
        by_level = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }

        for assessment in assessments:
            by_level[assessment.risk_level].append({
                "constraint": assessment.constraint_name,
                "headroom_percent": str(assessment.headroom_percent),
                "prob_binding": str(assessment.probability_of_binding),
                "prob_violation": str(assessment.probability_of_violation),
                "action": assessment.recommended_action
            })

        # Summary statistics
        total = len(assessments)
        summary = {
            "total_constraints": total,
            "critical_count": len(by_level["critical"]),
            "high_count": len(by_level["high"]),
            "medium_count": len(by_level["medium"]),
            "low_count": len(by_level["low"]),
            "overall_health": self._compute_overall_health(by_level)
        }

        # Top risks
        top_risks = [
            {
                "constraint": a.constraint_name,
                "risk_level": a.risk_level,
                "prob_binding": f"{float(a.probability_of_binding) * 100:.1f}%",
                "action": a.recommended_action
            }
            for a in assessments[:5]  # Top 5 by risk
        ]

        return {
            "summary": summary,
            "by_risk_level": by_level,
            "top_risks": top_risks,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _normal_cdf(self, x: Decimal) -> Decimal:
        """
        Standard normal CDF approximation - DETERMINISTIC.

        Uses Abramowitz and Stegun approximation.
        """
        x_float = float(x)

        # Handle extreme values
        if x_float < -5:
            return Decimal("0")
        if x_float > 5:
            return Decimal("1")

        # Constants for approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x_float >= 0 else -1
        x_abs = abs(x_float)

        t = 1.0 / (1.0 + p * x_abs)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x_abs * x_abs)

        result = 0.5 * (1.0 + sign * (2 * y - 1))
        return Decimal(str(result)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _compute_overall_health(self, by_level: Dict[str, List]) -> str:
        """Compute overall constraint health status - DETERMINISTIC."""
        if by_level["critical"]:
            return "critical"
        elif by_level["high"]:
            return "at_risk"
        elif by_level["medium"]:
            return "monitoring"
        else:
            return "healthy"


class ScenarioComparisonEngine:
    """
    Generate scenario comparison data - ZERO HALLUCINATION.

    Compares outcomes across scenarios for robust decision support.
    """

    def __init__(self):
        """Initialize scenario comparison engine."""
        pass

    def compare_scenarios(
        self,
        scenario_set: ScenarioSet,
        metric_name: str,
        compute_regret: bool = True
    ) -> ScenarioComparison:
        """
        Compare scenarios for a metric - DETERMINISTIC.

        Args:
            scenario_set: Set of scenarios
            metric_name: Metric to compare
            compute_regret: Whether to compute regret values

        Returns:
            ScenarioComparison with statistics
        """
        start_time = time.time()

        # Extract metric values
        values_by_scenario: Dict[str, Decimal] = {}
        unit = ""

        for scenario in scenario_set.scenarios:
            var = scenario.get_variable(metric_name)
            if var:
                values_by_scenario[scenario.name] = var.value
                unit = var.unit

        if not values_by_scenario:
            raise ValueError(f"Metric {metric_name} not found in scenarios")

        # Compute statistics
        values = list(values_by_scenario.values())
        probs = {
            s.name: s.probability
            for s in scenario_set.scenarios
            if s.name in values_by_scenario
        }

        # Weighted expected value
        expected = sum(
            values_by_scenario[name] * probs[name]
            for name in values_by_scenario
        )

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        # Variance and std dev
        variance = sum(
            probs[name] * (v - expected) ** 2
            for name, v in values_by_scenario.items()
        )
        std_dev = variance.sqrt() if variance > 0 else Decimal("0")

        # Regret values (for min-max regret)
        regret_by_scenario = None
        max_regret = None

        if compute_regret:
            # For each scenario, regret = value - best possible value
            # (assuming lower is better for cost-like metrics)
            best_value = min_val
            regret_by_scenario = {
                name: value - best_value
                for name, value in values_by_scenario.items()
            }
            max_regret = max(regret_by_scenario.values())

        computation_time_ms = (time.time() - start_time) * 1000
        provenance = ProvenanceRecord.create(
            calculation_type="scenario_comparison",
            inputs={
                "scenario_set_id": str(scenario_set.set_id),
                "metric_name": metric_name,
                "num_scenarios": len(values_by_scenario)
            },
            outputs={
                "expected_value": str(expected),
                "range": str(range_val),
                "std_dev": str(std_dev)
            },
            computation_time_ms=computation_time_ms
        )

        return ScenarioComparison(
            scenarios_compared=[s.scenario_id for s in scenario_set.scenarios],
            metric_name=metric_name,
            unit=unit,
            values_by_scenario={k: v for k, v in values_by_scenario.items()},
            expected_value=expected.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            min_value=min_val,
            max_value=max_val,
            range_value=range_val,
            std_dev=std_dev.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            regret_by_scenario=regret_by_scenario,
            max_regret=max_regret,
            provenance=provenance
        )

    def generate_comparison_table(
        self,
        scenario_set: ScenarioSet,
        metric_names: List[str]
    ) -> Dict[str, Any]:
        """
        Generate comparison table for multiple metrics - DETERMINISTIC.

        Returns data suitable for table display.
        """
        rows = []

        # Build rows by scenario
        for scenario in scenario_set.scenarios:
            row = {
                "scenario_name": scenario.name,
                "probability": str(scenario.probability),
                "is_base_case": scenario.is_base_case,
                "is_worst_case": scenario.is_worst_case,
                "metrics": {}
            }

            for metric_name in metric_names:
                var = scenario.get_variable(metric_name)
                if var:
                    row["metrics"][metric_name] = {
                        "value": str(var.value),
                        "unit": var.unit
                    }

            rows.append(row)

        # Compute column statistics
        columns = {}
        for metric_name in metric_names:
            try:
                comparison = self.compare_scenarios(scenario_set, metric_name, compute_regret=False)
                columns[metric_name] = {
                    "expected": str(comparison.expected_value),
                    "min": str(comparison.min_value),
                    "max": str(comparison.max_value),
                    "range": str(comparison.range_value),
                    "unit": comparison.unit
                }
            except ValueError:
                columns[metric_name] = None

        return {
            "scenarios": rows,
            "statistics": columns,
            "num_scenarios": len(rows),
            "num_metrics": len(metric_names)
        }
