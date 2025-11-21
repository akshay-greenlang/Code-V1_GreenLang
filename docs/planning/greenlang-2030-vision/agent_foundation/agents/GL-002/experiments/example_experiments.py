# -*- coding: utf-8 -*-
"""
Example Experiment Configurations for GL-002

This module provides ready-to-use experiment configurations for
common optimization scenarios.

Example:
    >>> from experiments.example_experiments import COMBUSTION_ALGORITHM_TEST
    >>> manager = ExperimentManager(db_url, redis_url)
    >>> await manager.initialize()
    >>> experiment = await manager.create_experiment(**COMBUSTION_ALGORITHM_TEST)
"""

from typing import Dict, Any, List
from .experiment_models import ExperimentVariant, MetricType


# ============================================================================
# EXPERIMENT 1: Combustion Optimization Algorithm Comparison
# ============================================================================

COMBUSTION_ALGORITHM_TEST: Dict[str, Any] = {
    "name": "combustion_algorithm_v2_test",
    "description": "Test new combustion optimization algorithm vs. current baseline",
    "hypothesis": "New ML-based combustion algorithm will improve energy savings by 15% while maintaining safety margins",
    "variants": [
        ExperimentVariant(
            name="control",
            traffic_split=0.5,
            config={
                "algorithm": "rule_based_v1",
                "o2_target_range": [3.0, 4.5],
                "efficiency_threshold": 85.0
            },
            description="Current rule-based combustion optimization",
            is_control=True
        ),
        ExperimentVariant(
            name="ml_algorithm",
            traffic_split=0.5,
            config={
                "algorithm": "ml_based_v2",
                "o2_target_range": [2.5, 4.0],
                "efficiency_threshold": 87.0,
                "ml_model": "gradient_boosting_v2.pkl"
            },
            description="New ML-based combustion optimization"
        )
    ],
    "primary_metric": "energy_savings_kwh",
    "primary_metric_type": MetricType.CONTINUOUS,
    "secondary_metrics": [
        "efficiency_improvement_percent",
        "co_emissions_ppm",
        "nox_emissions_ppm",
        "user_satisfaction_rating"
    ],
    "duration_days": 30,
    "created_by": "optimization_team",
    "tags": ["combustion", "ml", "energy-savings", "high-priority"]
}


# ============================================================================
# EXPERIMENT 2: Efficiency Calculation Method
# ============================================================================

EFFICIENCY_CALCULATION_TEST: Dict[str, Any] = {
    "name": "efficiency_calculation_method_test",
    "description": "Compare direct vs. indirect efficiency calculation methods",
    "hypothesis": "Indirect method (using flue gas analysis) provides more accurate efficiency measurements",
    "variants": [
        ExperimentVariant(
            name="control_direct",
            traffic_split=0.5,
            config={
                "method": "direct",
                "formula": "efficiency = (energy_output / fuel_input) * 100"
            },
            description="Direct efficiency calculation (energy in/out)",
            is_control=True
        ),
        ExperimentVariant(
            name="indirect_fluegas",
            traffic_split=0.5,
            config={
                "method": "indirect",
                "formula": "efficiency = 100 - losses",
                "losses_components": ["stack_loss", "radiation_loss", "blowdown_loss"]
            },
            "description": "Indirect method using flue gas analysis"
        )
    ],
    "primary_metric": "calculation_accuracy",
    "primary_metric_type": MetricType.CONTINUOUS,
    "secondary_metrics": [
        "calculation_time_ms",
        "data_quality_score",
        "user_trust_rating"
    ],
    "duration_days": 14,
    "created_by": "engineering_team",
    "tags": ["calculation", "accuracy", "methodology"]
}


# ============================================================================
# EXPERIMENT 3: Alert Threshold Optimization
# ============================================================================

ALERT_THRESHOLD_TEST: Dict[str, Any] = {
    "name": "alert_threshold_optimization_test",
    "description": "Optimize alert thresholds to reduce false positives while catching real issues",
    "hypothesis": "Increasing efficiency drop threshold from 2% to 3% will reduce false alerts by 40% without missing critical issues",
    "variants": [
        ExperimentVariant(
            name="control_strict",
            traffic_split=0.33,
            config={
                "efficiency_drop_threshold": 2.0,
                "consecutive_violations": 2,
                "alert_cooldown_minutes": 30
            },
            description="Current strict thresholds (2% drop)",
            is_control=True
        ),
        ExperimentVariant(
            name="moderate_threshold",
            traffic_split=0.33,
            config={
                "efficiency_drop_threshold": 3.0,
                "consecutive_violations": 2,
                "alert_cooldown_minutes": 30
            },
            description="Moderate threshold (3% drop)"
        ),
        ExperimentVariant(
            name="adaptive_threshold",
            traffic_split=0.34,
            config={
                "efficiency_drop_threshold": "adaptive",
                "consecutive_violations": 3,
                "alert_cooldown_minutes": 45,
                "baseline_window_hours": 24
            },
            description="Adaptive threshold based on historical baseline"
        )
    ],
    "primary_metric": "alert_precision",
    "primary_metric_type": MetricType.CONTINUOUS,
    "secondary_metrics": [
        "alert_recall",
        "false_positive_rate",
        "user_alert_fatigue_score",
        "time_to_resolution_minutes"
    ],
    "duration_days": 21,
    "created_by": "monitoring_team",
    "tags": ["alerting", "thresholds", "ux"]
}


# ============================================================================
# EXPERIMENT 4: UI/UX Improvement
# ============================================================================

UI_OPTIMIZATION_TEST: Dict[str, Any] = {
    "name": "optimization_detail_ui_test",
    "description": "Test simplified optimization detail UI vs. detailed technical view",
    "hypothesis": "Simplified UI with key metrics will increase user engagement and feedback submission rate",
    "variants": [
        ExperimentVariant(
            name="control_detailed",
            traffic_split=0.5,
            config={
                "ui_version": "detailed_v1",
                "show_technical_details": True,
                "charts": ["efficiency_trend", "fuel_consumption", "emissions", "cost_savings"],
                "metrics_count": 15
            },
            description="Current detailed UI with all technical metrics",
            is_control=True
        ),
        ExperimentVariant(
            name="simplified_ui",
            traffic_split=0.5,
            config={
                "ui_version": "simplified_v2",
                "show_technical_details": False,
                "charts": ["cost_savings", "efficiency_score"],
                "metrics_count": 5,
                "key_metrics": ["savings_amount", "efficiency_improvement", "payback_period"]
            },
            description="Simplified UI focusing on business impact"
        )
    ],
    "primary_metric": "feedback_submission_rate",
    "primary_metric_type": MetricType.CONVERSION_RATE,
    "secondary_metrics": [
        "page_engagement_time_seconds",
        "optimization_acceptance_rate",
        "user_satisfaction_rating",
        "detail_expansion_rate"
    ],
    "duration_days": 14,
    "created_by": "product_team",
    "tags": ["ui", "ux", "engagement", "feedback"]
}


# ============================================================================
# EXPERIMENT 5: Recommendation Frequency
# ============================================================================

RECOMMENDATION_FREQUENCY_TEST: Dict[str, Any] = {
    "name": "recommendation_frequency_test",
    "description": "Optimize frequency of optimization recommendations",
    "hypothesis": "Daily recommendations will increase implementation rate vs. weekly batch recommendations",
    "variants": [
        ExperimentVariant(
            name="control_weekly",
            traffic_split=0.5,
            config={
                "frequency": "weekly",
                "day_of_week": "monday",
                "batch_size": "all_available",
                "prioritization": "roi_descending"
            },
            description="Weekly batch recommendations (current)",
            is_control=True
        ),
        ExperimentVariant(
            name="daily_top3",
            traffic_split=0.5,
            config={
                "frequency": "daily",
                "time_of_day": "09:00",
                "batch_size": 3,
                "prioritization": "quick_wins"
            },
            description="Daily top 3 recommendations"
        )
    ],
    "primary_metric": "implementation_rate",
    "primary_metric_type": MetricType.CONVERSION_RATE,
    "secondary_metrics": [
        "time_to_implementation_hours",
        "total_savings_realized_kwh",
        "user_overwhelm_score",
        "recommendation_fatigue_rate"
    ],
    "duration_days": 28,
    "created_by": "optimization_team",
    "tags": ["recommendations", "frequency", "conversion"]
}


# ============================================================================
# EXPERIMENT 6: Predictive Model Comparison
# ============================================================================

PREDICTIVE_MODEL_TEST: Dict[str, Any] = {
    "name": "savings_prediction_model_test",
    "description": "Compare linear regression vs. gradient boosting for savings prediction",
    "hypothesis": "Gradient boosting model will improve prediction accuracy by 20% over linear regression",
    "variants": [
        ExperimentVariant(
            name="control_linear",
            traffic_split=0.5,
            config={
                "model_type": "linear_regression",
                "features": ["efficiency_current", "fuel_type", "load_factor", "age_years"],
                "model_path": "models/linear_regression_v1.pkl"
            },
            description="Linear regression model (baseline)",
            is_control=True
        ),
        ExperimentVariant(
            name="gradient_boosting",
            traffic_split=0.5,
            config={
                "model_type": "gradient_boosting",
                "features": [
                    "efficiency_current", "fuel_type", "load_factor", "age_years",
                    "maintenance_score", "operating_hours", "seasonal_factor"
                ],
                "model_path": "models/gradient_boosting_v2.pkl",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1
                }
            },
            description="Gradient boosting model with extended features"
        )
    ],
    "primary_metric": "prediction_mae",
    "primary_metric_type": MetricType.CONTINUOUS,
    "secondary_metrics": [
        "prediction_mape",
        "within_10pct_accuracy_rate",
        "inference_time_ms",
        "model_confidence_score"
    ],
    "duration_days": 45,
    "created_by": "ml_team",
    "tags": ["ml", "prediction", "accuracy", "models"]
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_all_experiments() -> List[Dict[str, Any]]:
    """
    Get list of all example experiments.

    Returns:
        List of experiment configurations
    """
    return [
        COMBUSTION_ALGORITHM_TEST,
        EFFICIENCY_CALCULATION_TEST,
        ALERT_THRESHOLD_TEST,
        UI_OPTIMIZATION_TEST,
        RECOMMENDATION_FREQUENCY_TEST,
        PREDICTIVE_MODEL_TEST
    ]


def get_experiment_by_name(name: str) -> Dict[str, Any]:
    """
    Get experiment configuration by name.

    Args:
        name: Experiment name

    Returns:
        Experiment configuration

    Raises:
        ValueError: If experiment not found
    """
    experiments = {
        exp["name"]: exp
        for exp in get_all_experiments()
    }

    if name not in experiments:
        raise ValueError(f"Experiment '{name}' not found")

    return experiments[name]


def get_experiments_by_tag(tag: str) -> List[Dict[str, Any]]:
    """
    Get experiments filtered by tag.

    Args:
        tag: Tag to filter by

    Returns:
        List of matching experiment configurations
    """
    return [
        exp for exp in get_all_experiments()
        if tag in exp.get("tags", [])
    ]


# ============================================================================
# Experiment Templates
# ============================================================================

def create_algorithm_comparison_template(
    algorithm_a: str,
    algorithm_b: str,
    config_a: Dict[str, Any],
    config_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create template for algorithm comparison experiment.

    Args:
        algorithm_a: Name of control algorithm
        algorithm_b: Name of treatment algorithm
        config_a: Configuration for algorithm A
        config_b: Configuration for algorithm B

    Returns:
        Experiment configuration dictionary
    """
    return {
        "name": f"{algorithm_a}_vs_{algorithm_b}_test",
        "description": f"Compare {algorithm_a} vs {algorithm_b} algorithms",
        "hypothesis": f"{algorithm_b} will outperform {algorithm_a}",
        "variants": [
            ExperimentVariant(
                name=f"control_{algorithm_a}",
                traffic_split=0.5,
                config=config_a,
                description=f"{algorithm_a} (control)",
                is_control=True
            ),
            ExperimentVariant(
                name=f"treatment_{algorithm_b}",
                traffic_split=0.5,
                config=config_b,
                description=f"{algorithm_b} (treatment)"
            )
        ],
        "primary_metric": "algorithm_performance",
        "primary_metric_type": MetricType.CONTINUOUS,
        "secondary_metrics": ["execution_time_ms", "accuracy", "user_satisfaction"],
        "duration_days": 21,
        "created_by": "engineering_team",
        "tags": ["algorithm", "comparison"]
    }
