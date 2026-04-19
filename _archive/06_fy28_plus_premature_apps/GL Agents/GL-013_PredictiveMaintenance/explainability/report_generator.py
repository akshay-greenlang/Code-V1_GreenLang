# -*- coding: utf-8 -*-
"""
Report Generator for GL-013 PredictiveMaintenance Explainability Module.

Generates explanation reports in multiple formats:
- Per-decision explanation reports
- Batch explanation summaries
- Dashboard-ready data formats
- JSON exports for API responses
- PDF-ready structured data

Zero-hallucination guarantees:
- All numeric values come from deterministic calculations
- Provenance hashing for audit trails
- No LLM-generated numbers

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import numpy as np

from .explanation_schemas import (
    MaintenanceExplanationReport,
    FeatureContribution,
    SHAPExplanation,
    LIMEExplanation,
    AttentionExplanation,
    CausalExplanation,
    RootCauseHypothesis,
    ConfidenceBounds,
    UncertaintyRange,
    PredictionType,
    ConfidenceLevel,
    DashboardExplanationData,
)

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    include_raw_data: bool = True
    include_visualizations: bool = True
    max_features_display: int = 10
    decimal_precision: int = 4
    include_timestamps: bool = True
    include_provenance: bool = True
    format_numbers: bool = True
    generate_summary: bool = True


@dataclass
class ReportMetadata:
    """Metadata for generated reports."""

    report_id: str
    generation_timestamp: datetime
    generation_time_ms: float
    report_type: str
    version: str = "1.0.0"
    generator: str = "GL-013 PredictiveMaintenance Explainability Module"
    deterministic: bool = True
    provenance_hash: str = ""


class ReportGenerator:
    """
    Generates explanation reports for GL-013 PredictiveMaintenance.

    Produces structured reports suitable for:
    - Dashboard visualization
    - API responses
    - Audit trails
    - PDF generation (data structure only)
    - Regulatory compliance documentation
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.

        Args:
            config: Report configuration settings
        """
        self.config = config or ReportConfig()
        self._report_history: List[ReportMetadata] = []

        logger.info("ReportGenerator initialized")

    def generate_decision_report(
        self,
        explanation_report: ExplanationReport,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a per-decision explanation report.

        Args:
            explanation_report: The explanation report to format
            additional_context: Additional context to include

        Returns:
            Formatted report dictionary
        """
        start_time = time.time()

        report = {
            "report_type": "decision_explanation",
            "metadata": self._generate_metadata("decision_explanation", explanation_report.report_id),
            "summary": self._generate_summary(explanation_report),
            "prediction": self._format_prediction(explanation_report),
            "explanations": self._format_explanations(explanation_report),
            "uncertainty": self._format_uncertainty(explanation_report.uncertainty),
            "top_features": self._format_feature_contributions(
                explanation_report.top_features,
                max_features=self.config.max_features_display
            ),
        }

        if explanation_report.counterfactuals:
            report["counterfactuals"] = self._format_counterfactuals(
                explanation_report.counterfactuals
            )

        if additional_context:
            report["context"] = additional_context

        if self.config.include_provenance:
            report["provenance"] = {
                "hash": explanation_report.provenance_hash,
                "deterministic": explanation_report.deterministic,
                "timestamp": explanation_report.timestamp.isoformat()
            }

        # Update metadata with generation time
        elapsed_ms = (time.time() - start_time) * 1000
        report["metadata"]["generation_time_ms"] = elapsed_ms

        # Track report
        self._track_report(report["metadata"])

        return report

    def generate_batch_report(
        self,
        summary: BatchExplanationSummary,
        individual_reports: Optional[List[ExplanationReport]] = None
    ) -> Dict[str, Any]:
        """
        Generate a batch explanation summary report.

        Args:
            summary: Batch summary statistics
            individual_reports: Optional list of individual reports

        Returns:
            Formatted batch report dictionary
        """
        start_time = time.time()

        report = {
            "report_type": "batch_explanation_summary",
            "metadata": self._generate_metadata("batch_summary", summary.summary_id),
            "batch_statistics": {
                "batch_size": summary.batch_size,
                "prediction_type": summary.prediction_type.value,
                "mean_prediction": self._format_number(summary.mean_prediction),
                "std_prediction": self._format_number(summary.std_prediction),
                "mean_confidence": self._format_number(summary.mean_confidence),
            },
            "global_feature_importance": self._format_feature_importance(
                summary.global_feature_importance,
                summary.feature_importance_std
            ),
            "quality_metrics": {
                "mean_shap_consistency": self._format_number(summary.mean_shap_consistency),
                "mean_lime_r2": self._format_number(summary.mean_lime_r2),
            }
        }

        if summary.common_binding_constraints:
            report["common_binding_constraints"] = summary.common_binding_constraints

        if individual_reports and self.config.include_raw_data:
            report["individual_summaries"] = [
                {
                    "report_id": r.report_id,
                    "prediction": self._format_number(r.prediction_value),
                    "confidence": r.confidence_level.value
                }
                for r in individual_reports[:10]  # Limit to first 10
            ]

        elapsed_ms = (time.time() - start_time) * 1000
        report["metadata"]["generation_time_ms"] = elapsed_ms

        self._track_report(report["metadata"])

        return report

    def generate_dashboard_data(
        self,
        explanation_report: ExplanationReport
    ) -> DashboardExplanationData:
        """
        Generate dashboard-ready data for visualization.

        Args:
            explanation_report: The explanation report to convert

        Returns:
            DashboardExplanationData with pre-computed visualization data
        """
        # Generate waterfall chart data
        waterfall_data = self._generate_waterfall_data(explanation_report)

        # Generate force plot data
        force_plot_data = self._generate_force_plot_data(explanation_report)

        # Generate feature importance chart data
        feature_importance_chart = self._generate_feature_importance_chart(
            explanation_report.top_features
        )

        # Generate counterfactual chart if available
        counterfactual_chart = None
        if explanation_report.counterfactuals:
            counterfactual_chart = self._generate_counterfactual_chart(
                explanation_report.counterfactuals
            )

        return DashboardExplanationData(
            waterfall_data=waterfall_data,
            force_plot_data=force_plot_data,
            feature_importance_chart=feature_importance_chart,
            summary_plot_data=None,  # Requires batch data
            importance_over_time=None,  # Requires historical data
            counterfactual_chart=counterfactual_chart,
            timestamp=datetime.utcnow()
        )

    def generate_dashboard_data_batch(
        self,
        reports: List[ExplanationReport]
    ) -> Dict[str, Any]:
        """
        Generate dashboard data from multiple reports.

        Args:
            reports: List of explanation reports

        Returns:
            Dashboard data with summary and time series
        """
        if not reports:
            return {"error": "No reports provided"}

        # Generate summary plot data (like SHAP beeswarm)
        summary_plot_data = self._generate_summary_plot_data(reports)

        # Generate feature importance over time
        importance_over_time = self._generate_importance_over_time(reports)

        # Generate prediction trend
        prediction_trend = [
            {
                "timestamp": r.timestamp.isoformat(),
                "prediction": r.prediction_value,
                "confidence": r.confidence_level.value
            }
            for r in sorted(reports, key=lambda x: x.timestamp)
        ]

        return {
            "summary_plot_data": summary_plot_data,
            "importance_over_time": importance_over_time,
            "prediction_trend": prediction_trend,
            "batch_size": len(reports),
            "timestamp": datetime.utcnow().isoformat()
        }

    def generate_optimization_report(
        self,
        decision_explanation: DecisionExplanation,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate optimization decision explanation report.

        Args:
            decision_explanation: The decision explanation
            context: Additional optimization context

        Returns:
            Formatted optimization report
        """
        start_time = time.time()

        report = {
            "report_type": "optimization_decision",
            "metadata": self._generate_metadata(
                "optimization_decision",
                decision_explanation.decision_id
            ),
            "objective": {
                "value": self._format_number(decision_explanation.objective_value),
                "optimality_gap": self._format_number(decision_explanation.optimality_gap),
            },
            "binding_constraints": {
                "count": len(decision_explanation.binding_constraints),
                "constraints": decision_explanation.binding_constraints
            },
            "shadow_prices": {
                name: self._format_number(value)
                for name, value in decision_explanation.shadow_prices.items()
            },
            "reduced_costs": {
                name: self._format_number(value)
                for name, value in decision_explanation.reduced_costs.items()
            },
            "sensitivity_analysis": self._format_sensitivity(
                decision_explanation.sensitivity_analysis
            ),
        }

        if decision_explanation.alternative_solutions:
            report["alternatives"] = self._format_alternatives(
                decision_explanation.alternative_solutions
            )

        if context:
            report["context"] = context

        elapsed_ms = (time.time() - start_time) * 1000
        report["metadata"]["generation_time_ms"] = elapsed_ms

        self._track_report(report["metadata"])

        return report

    def export_to_json(
        self,
        report: Dict[str, Any],
        indent: int = 2
    ) -> str:
        """
        Export report to JSON string.

        Args:
            report: Report dictionary
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(report, indent=indent, default=str)

    def generate_audit_record(
        self,
        explanation_report: ExplanationReport
    ) -> Dict[str, Any]:
        """
        Generate audit record for regulatory compliance.

        Args:
            explanation_report: The explanation report

        Returns:
            Audit record with provenance information
        """
        return {
            "audit_type": "ml_explanation",
            "report_id": explanation_report.report_id,
            "timestamp": explanation_report.timestamp.isoformat(),
            "model_info": {
                "name": explanation_report.model_name,
                "version": explanation_report.model_version
            },
            "prediction": {
                "type": explanation_report.prediction_type.value,
                "value": explanation_report.prediction_value,
                "confidence": explanation_report.confidence_level.value
            },
            "determinism": {
                "is_deterministic": explanation_report.deterministic,
                "provenance_hash": explanation_report.provenance_hash
            },
            "input_hash": hashlib.sha256(
                str(explanation_report.input_features).encode()
            ).hexdigest(),
            "computation_time_ms": explanation_report.computation_time_ms
        }

    def _generate_metadata(
        self,
        report_type: str,
        report_id: str
    ) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            "report_id": report_id,
            "report_type": report_type,
            "generation_timestamp": datetime.utcnow().isoformat(),
            "generator": "GL-013 PredictiveMaintenance Explainability Module",
            "version": "1.0.0",
            "deterministic": True
        }

    def _generate_summary(
        self,
        report: ExplanationReport
    ) -> Dict[str, Any]:
        """Generate human-readable summary."""
        top_feature = report.top_features[0] if report.top_features else None

        return {
            "prediction_type": report.prediction_type.value,
            "prediction_value": self._format_number(report.prediction_value),
            "confidence": report.confidence_level.value,
            "top_driver": top_feature.feature_name if top_feature else "N/A",
            "top_driver_impact": self._format_number(top_feature.contribution) if top_feature else 0,
            "narrative": report.narrative_summary
        }

    def _format_prediction(
        self,
        report: ExplanationReport
    ) -> Dict[str, Any]:
        """Format prediction data."""
        return {
            "value": self._format_number(report.prediction_value),
            "type": report.prediction_type.value,
            "confidence_level": report.confidence_level.value,
            "input_features": {
                k: self._format_number(v)
                for k, v in report.input_features.items()
            }
        }

    def _format_explanations(
        self,
        report: ExplanationReport
    ) -> Dict[str, Any]:
        """Format explanation data."""
        explanations = {}

        if report.shap_explanation:
            explanations["shap"] = {
                "base_value": self._format_number(report.shap_explanation.base_value),
                "consistency_check": self._format_number(
                    report.shap_explanation.consistency_check
                ),
                "explainer_type": report.shap_explanation.explainer_type,
                "feature_contributions": self._format_feature_contributions(
                    report.shap_explanation.feature_contributions
                )
            }

        if report.lime_explanation:
            explanations["lime"] = {
                "local_model_r2": self._format_number(report.lime_explanation.local_model_r2),
                "local_model_intercept": self._format_number(
                    report.lime_explanation.local_model_intercept
                ),
                "neighborhood_size": report.lime_explanation.neighborhood_size,
                "kernel_width": self._format_number(report.lime_explanation.kernel_width),
                "feature_contributions": self._format_feature_contributions(
                    report.lime_explanation.feature_contributions
                )
            }

        if report.decision_explanation:
            explanations["optimization"] = {
                "objective_value": self._format_number(
                    report.decision_explanation.objective_value
                ),
                "binding_constraints": report.decision_explanation.binding_constraints,
                "optimality_gap": self._format_number(
                    report.decision_explanation.optimality_gap
                )
            }

        return explanations

    def _format_uncertainty(
        self,
        uncertainty: UncertaintyRange
    ) -> Dict[str, Any]:
        """Format uncertainty data."""
        return {
            "point_estimate": self._format_number(uncertainty.point_estimate),
            "standard_error": self._format_number(uncertainty.standard_error),
            "confidence_interval": {
                "lower": self._format_number(uncertainty.confidence_interval.lower_bound),
                "upper": self._format_number(uncertainty.confidence_interval.upper_bound),
                "level": uncertainty.confidence_interval.confidence_level,
                "method": uncertainty.confidence_interval.method
            },
            "variance": self._format_number(uncertainty.prediction_variance),
            "epistemic": self._format_number(uncertainty.epistemic_uncertainty),
            "aleatoric": self._format_number(uncertainty.aleatoric_uncertainty),
            "total": self._format_number(uncertainty.total_uncertainty)
        }

    def _format_feature_contributions(
        self,
        contributions: List[FeatureContribution],
        max_features: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Format feature contributions."""
        max_features = max_features or self.config.max_features_display

        return [
            {
                "feature": c.feature_name,
                "value": self._format_number(c.feature_value),
                "contribution": self._format_number(c.contribution),
                "percentage": self._format_number(c.contribution_percentage),
                "direction": c.direction,
                "unit": c.unit
            }
            for c in contributions[:max_features]
        ]

    def _format_counterfactuals(
        self,
        counterfactuals: List[Counterfactual]
    ) -> List[Dict[str, Any]]:
        """Format counterfactual data."""
        return [
            {
                "id": cf.counterfactual_id,
                "target_prediction": self._format_number(cf.target_prediction),
                "original_prediction": self._format_number(cf.original_prediction),
                "feature_changes": {
                    name: {
                        "from": self._format_number(change["from"]),
                        "to": self._format_number(change["to"]),
                        "change": self._format_number(change.get("change", 0))
                    }
                    for name, change in cf.feature_changes.items()
                },
                "feasibility": self._format_number(cf.feasibility_score),
                "sparsity": cf.sparsity,
                "distance": self._format_number(cf.distance),
                "valid": cf.validity,
                "actionable": cf.is_actionable
            }
            for cf in counterfactuals
        ]

    def _format_feature_importance(
        self,
        importance: Dict[str, float],
        std: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Format global feature importance with uncertainty."""
        return [
            {
                "feature": name,
                "importance": self._format_number(value),
                "std": self._format_number(std.get(name, 0))
            }
            for name, value in sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.config.max_features_display]
        ]

    def _format_sensitivity(
        self,
        sensitivity: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Format sensitivity analysis."""
        return {
            name: {
                k: self._format_number(v)
                for k, v in analysis.items()
            }
            for name, analysis in sensitivity.items()
        }

    def _format_alternatives(
        self,
        alternatives: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format alternative solutions."""
        return [
            {
                "objective_value": self._format_number(alt.get("objective_value", 0)),
                "gap_from_optimal": self._format_number(alt.get("gap_from_optimal", 0)),
                "changed_variable": alt.get("changed_variable", ""),
                "change_amount": self._format_number(alt.get("change_amount", 0))
            }
            for alt in alternatives[:5]
        ]

    def _generate_waterfall_data(
        self,
        report: ExplanationReport
    ) -> List[Dict[str, Any]]:
        """Generate waterfall chart data."""
        data = []

        # Base value
        base_value = 0
        if report.shap_explanation:
            base_value = report.shap_explanation.base_value
        elif report.lime_explanation:
            base_value = report.lime_explanation.local_model_intercept

        data.append({
            "name": "Base Value",
            "value": base_value,
            "cumulative": base_value,
            "type": "base"
        })

        # Feature contributions
        cumulative = base_value
        for contrib in report.top_features[:self.config.max_features_display]:
            cumulative += contrib.contribution
            data.append({
                "name": contrib.feature_name,
                "value": contrib.contribution,
                "cumulative": cumulative,
                "type": "positive" if contrib.contribution >= 0 else "negative",
                "feature_value": contrib.feature_value
            })

        # Final prediction
        data.append({
            "name": "Prediction",
            "value": report.prediction_value,
            "cumulative": report.prediction_value,
            "type": "total"
        })

        return data

    def _generate_force_plot_data(
        self,
        report: ExplanationReport
    ) -> Dict[str, Any]:
        """Generate force plot data."""
        positive = []
        negative = []

        for contrib in report.top_features:
            entry = {
                "feature": contrib.feature_name,
                "value": contrib.feature_value,
                "effect": abs(contrib.contribution)
            }
            if contrib.contribution >= 0:
                positive.append(entry)
            else:
                negative.append(entry)

        base_value = 0
        if report.shap_explanation:
            base_value = report.shap_explanation.base_value
        elif report.lime_explanation:
            base_value = report.lime_explanation.local_model_intercept

        return {
            "base_value": base_value,
            "prediction": report.prediction_value,
            "positive_features": positive,
            "negative_features": negative,
            "positive_sum": sum(f["effect"] for f in positive),
            "negative_sum": sum(f["effect"] for f in negative)
        }

    def _generate_feature_importance_chart(
        self,
        contributions: List[FeatureContribution]
    ) -> List[Dict[str, Any]]:
        """Generate feature importance bar chart data."""
        return [
            {
                "feature": c.feature_name,
                "importance": abs(c.contribution),
                "direction": c.direction,
                "percentage": c.contribution_percentage
            }
            for c in contributions[:self.config.max_features_display]
        ]

    def _generate_counterfactual_chart(
        self,
        counterfactuals: List[Counterfactual]
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual comparison chart data."""
        return [
            {
                "id": cf.counterfactual_id,
                "sparsity": cf.sparsity,
                "distance": cf.distance,
                "feasibility": cf.feasibility_score,
                "prediction_change": cf.target_prediction - cf.original_prediction,
                "changes": list(cf.feature_changes.keys())
            }
            for cf in counterfactuals
        ]

    def _generate_summary_plot_data(
        self,
        reports: List[ExplanationReport]
    ) -> List[Dict[str, Any]]:
        """Generate summary plot data (like SHAP beeswarm)."""
        # Collect all feature contributions across reports
        feature_data: Dict[str, List[Dict[str, float]]] = {}

        for report in reports:
            for contrib in report.top_features:
                if contrib.feature_name not in feature_data:
                    feature_data[contrib.feature_name] = []
                feature_data[contrib.feature_name].append({
                    "value": contrib.feature_value,
                    "contribution": contrib.contribution
                })

        # Format for visualization
        summary_data = []
        for feature_name, data_points in feature_data.items():
            summary_data.append({
                "feature": feature_name,
                "data_points": data_points,
                "mean_contribution": np.mean([d["contribution"] for d in data_points]),
                "std_contribution": np.std([d["contribution"] for d in data_points])
            })

        # Sort by mean absolute contribution
        summary_data.sort(key=lambda x: abs(x["mean_contribution"]), reverse=True)

        return summary_data[:self.config.max_features_display]

    def _generate_importance_over_time(
        self,
        reports: List[ExplanationReport]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate feature importance trends over time."""
        # Sort reports by timestamp
        sorted_reports = sorted(reports, key=lambda x: x.timestamp)

        importance_time_series: Dict[str, List[Dict[str, Any]]] = {}

        for report in sorted_reports:
            for contrib in report.top_features[:5]:  # Top 5 features
                if contrib.feature_name not in importance_time_series:
                    importance_time_series[contrib.feature_name] = []

                importance_time_series[contrib.feature_name].append({
                    "timestamp": report.timestamp.isoformat(),
                    "importance": abs(contrib.contribution),
                    "contribution": contrib.contribution
                })

        return importance_time_series

    def _format_number(self, value: Union[int, float]) -> Union[int, float]:
        """Format number with configured precision."""
        if not self.config.format_numbers:
            return value
        if isinstance(value, int):
            return value
        return round(value, self.config.decimal_precision)

    def _track_report(self, metadata: Dict[str, Any]) -> None:
        """Track generated report."""
        self._report_history.append(ReportMetadata(
            report_id=metadata["report_id"],
            generation_timestamp=datetime.fromisoformat(metadata["generation_timestamp"]),
            generation_time_ms=metadata.get("generation_time_ms", 0),
            report_type=metadata["report_type"],
            version=metadata.get("version", "1.0.0"),
            generator=metadata.get("generator", "Unknown"),
            deterministic=metadata.get("deterministic", True)
        ))

        # Keep only last 1000 reports
        if len(self._report_history) > 1000:
            self._report_history = self._report_history[-1000:]

    def get_report_history(
        self,
        report_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get report generation history."""
        history = self._report_history

        if report_type:
            history = [r for r in history if r.report_type == report_type]

        return [
            {
                "report_id": r.report_id,
                "type": r.report_type,
                "timestamp": r.generation_timestamp.isoformat(),
                "generation_time_ms": r.generation_time_ms
            }
            for r in history[-limit:]
        ]


# Convenience functions

def generate_quick_report(
    explanation_report: ExplanationReport,
    include_raw: bool = False
) -> Dict[str, Any]:
    """
    Quick function to generate a basic report.

    Args:
        explanation_report: Explanation report
        include_raw: Include raw data

    Returns:
        Formatted report
    """
    config = ReportConfig(include_raw_data=include_raw)
    generator = ReportGenerator(config)
    return generator.generate_decision_report(explanation_report)


def export_report_json(
    explanation_report: ExplanationReport,
    filepath: Optional[str] = None
) -> str:
    """
    Export explanation report to JSON.

    Args:
        explanation_report: Explanation report
        filepath: Optional file path to write to

    Returns:
        JSON string
    """
    generator = ReportGenerator()
    report = generator.generate_decision_report(explanation_report)
    json_str = generator.export_to_json(report)

    if filepath:
        with open(filepath, 'w') as f:
            f.write(json_str)

    return json_str


def export_report_html(
    report: MaintenanceExplanationReport,
    filepath: Optional[str] = None
) -> str:
    """
    Export explanation report to HTML.

    Args:
        report: MaintenanceExplanationReport to export
        filepath: Optional file path to write to

    Returns:
        HTML string
    """
    html_parts = [
        '<!DOCTYPE html>',
        '<html><head>',
        '<title>Maintenance Explanation Report</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; margin: 20px; }',
        'h1 { color: #2c3e50; }',
        'h2 { color: #34495e; border-bottom: 2px solid #3498db; }',
        '.feature { padding: 5px; margin: 5px 0; background: #ecf0f1; }',
        '.positive { border-left: 4px solid #27ae60; }',
        '.negative { border-left: 4px solid #e74c3c; }',
        '.root-cause { padding: 10px; margin: 10px 0; background: #ffeaa7; }',
        '</style>',
        '</head><body>',
        f'<h1>Predictive Maintenance Explanation Report</h1>',
        f'<p><strong>Report ID:</strong> {report.report_id}</p>',
        f'<p><strong>Asset ID:</strong> {report.asset_id}</p>',
        f'<p><strong>Prediction Type:</strong> {report.prediction_type.value}</p>',
        f'<p><strong>Prediction Value:</strong> {report.prediction_value:.4f}</p>',
        f'<p><strong>Confidence:</strong> {report.confidence_level.value}</p>',
        '<h2>Top Contributing Features</h2>',
    ]
    
    for f in report.top_features[:10]:
        css_class = "positive" if f.direction == "positive" else "negative"
        html_parts.append(
            f'<div class="feature {css_class}">'
            f'<strong>{f.feature_name}</strong>: {f.contribution:.4f} '
            f'({f.contribution_percentage:.1f}%)</div>'
        )
    
    if report.top_root_causes:
        html_parts.append('<h2>Root Cause Analysis</h2>')
        for rc in report.top_root_causes[:5]:
            html_parts.append(
                f'<div class="root-cause">'
                f'<strong>#{rc.rank} {rc.cause_variable}</strong>: '
                f'Effect = {rc.causal_effect:.4f}, '
                f'Evidence Strength = {rc.evidence_strength:.2f}</div>'
            )
    
    html_parts.extend([
        '<h2>Metadata</h2>',
        f'<p><strong>Generated:</strong> {report.timestamp.isoformat()}</p>',
        f'<p><strong>Provenance Hash:</strong> {report.provenance_hash}</p>',
        f'<p><strong>Data Quality Score:</strong> {report.data_quality_score:.2f}</p>',
        '</body></html>'
    ])
    
    html_str = chr(10).join(html_parts)
    
    if filepath:
        with open(filepath, "w") as f:
            f.write(html_str)
    
    return html_str
