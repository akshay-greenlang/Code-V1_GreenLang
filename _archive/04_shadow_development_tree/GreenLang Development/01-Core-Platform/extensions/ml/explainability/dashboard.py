# -*- coding: utf-8 -*-
"""
ML Explainability Dashboard Component for GreenLang.

This module provides comprehensive dashboard functionality for visualizing
and managing ML model explanations. It includes data models for frontend
consumption, API endpoints for dashboard data retrieval, and state management
for user preferences and caching.

TASK-030: Explanation Dashboard Component

Features:
    - Dashboard Data Models (Pydantic) for JSON-serializable responses
    - FastAPI endpoints for dashboard data retrieval
    - Visualization data generators (SHAP waterfall, force plot, bar chart)
    - Dashboard state management (model tracking, cache, preferences)
    - Export functionality (JSON, CSV)
    - Provenance hashing for audit trails

Endpoints:
    GET /dashboard/explanation/{model_id} - Get explanation visualization data
    GET /dashboard/feature-importance/{model_id} - Get feature importance chart data
    GET /dashboard/counterfactuals/{model_id} - Get counterfactual comparison data
    GET /dashboard/history/{model_id} - Get prediction history with explanations
    GET /dashboard/summary - Get overall explainability summary

Example:
    >>> from greenlang.ml.explainability.dashboard import dashboard_router
    >>> # Include in FastAPI app
    >>> app.include_router(dashboard_router, prefix="/api/v1")

Author: GreenLang Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union
from uuid import uuid4

import numpy as np

# Conditional Pydantic import
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore
    Field = lambda *args, **kwargs: None  # type: ignore

# Conditional FastAPI import
try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore
    HTTPException = Exception  # type: ignore
    Query = lambda *args, **kwargs: None  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ChartType(str, Enum):
    """Supported chart types for visualizations."""
    WATERFALL = "waterfall"
    FORCE_PLOT = "force_plot"
    BAR_CHART = "bar_chart"
    BEESWARM = "beeswarm"
    HEATMAP = "heatmap"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    PIE_CHART = "pie_chart"
    HISTOGRAM = "histogram"


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"


class TimeRange(str, Enum):
    """Predefined time ranges for history queries."""
    LAST_HOUR = "1h"
    LAST_DAY = "24h"
    LAST_WEEK = "7d"
    LAST_MONTH = "30d"
    LAST_QUARTER = "90d"
    CUSTOM = "custom"


class DashboardViewMode(str, Enum):
    """Dashboard view modes."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPARISON = "comparison"
    HISTORICAL = "historical"


# =============================================================================
# PYDANTIC DATA MODELS
# =============================================================================

class FeatureContributionData(BaseModel):
    """Individual feature contribution for charts."""

    feature_name: str = Field(..., description="Name of the feature")
    display_name: str = Field(..., description="Human-readable display name")
    value: float = Field(..., description="Feature value in the instance")
    contribution: float = Field(..., description="SHAP/attribution value")
    percentage: float = Field(..., description="Percentage of total contribution")
    direction: str = Field(..., description="positive or negative impact")
    color: str = Field(default="#1976d2", description="Color for visualization")
    rank: int = Field(..., ge=1, description="Rank by importance")

    model_config = ConfigDict(
        json_encoders={float: lambda v: round(v, 6) if isinstance(v, float) else v}
    )


class FeatureContributionChart(BaseModel):
    """
    Data structure for feature contribution visualizations.

    Suitable for waterfall charts, bar charts, and force plots.
    """

    chart_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    chart_type: ChartType = Field(default=ChartType.WATERFALL)
    title: str = Field(default="Feature Contributions")
    subtitle: Optional[str] = Field(default=None)

    # Core data
    base_value: float = Field(..., description="Expected/baseline value")
    prediction: float = Field(..., description="Final prediction value")
    contributions: List[FeatureContributionData] = Field(
        default_factory=list,
        description="List of feature contributions"
    )

    # Cumulative values for waterfall
    cumulative_values: List[float] = Field(
        default_factory=list,
        description="Cumulative values for waterfall chart"
    )

    # Metadata
    model_id: str = Field(..., description="Model identifier")
    instance_id: Optional[str] = Field(default=None, description="Instance identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    # Visualization options
    show_values: bool = Field(default=True)
    show_percentages: bool = Field(default=True)
    max_features: int = Field(default=10, ge=1, le=50)
    color_positive: str = Field(default="#4caf50")
    color_negative: str = Field(default="#f44336")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None,
            float: lambda v: round(v, 6) if isinstance(v, float) else v
        }
    )


class GlobalImportanceChart(BaseModel):
    """
    Data structure for global feature importance visualizations.

    Shows overall model-level feature importance across all predictions.
    """

    chart_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    chart_type: ChartType = Field(default=ChartType.BAR_CHART)
    title: str = Field(default="Global Feature Importance")
    subtitle: Optional[str] = Field(default=None)

    # Core data
    features: List[str] = Field(..., description="Feature names in order")
    display_names: List[str] = Field(..., description="Human-readable names")
    importance_scores: List[float] = Field(..., description="Importance scores")
    importance_std: Optional[List[float]] = Field(
        default=None,
        description="Standard deviation of importance (if available)"
    )

    # Summary statistics
    total_features: int = Field(..., ge=1)
    sample_size: int = Field(..., ge=1, description="Samples used for computation")
    top_feature: str = Field(..., description="Most important feature")
    top_importance: float = Field(..., description="Importance of top feature")

    # Interaction effects (optional)
    interactions: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Feature interaction strengths"
    )

    # Metadata
    model_id: str = Field(..., description="Model identifier")
    method: str = Field(default="shap", description="Importance method used")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = Field(default="")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None,
            float: lambda v: round(v, 6) if isinstance(v, float) else v
        }
    )


class CounterfactualChange(BaseModel):
    """Single feature change in a counterfactual."""

    feature_name: str = Field(..., description="Feature name")
    display_name: str = Field(..., description="Human-readable name")
    original_value: float = Field(..., description="Original value")
    counterfactual_value: float = Field(..., description="New value")
    change_amount: float = Field(..., description="Absolute change")
    change_percentage: float = Field(..., description="Percentage change")
    direction: str = Field(..., description="increase or decrease")
    feasibility: float = Field(..., ge=0, le=1, description="Feasibility score")
    units: Optional[str] = Field(default=None, description="Feature units")


class CounterfactualComparisonView(BaseModel):
    """
    Data structure for counterfactual comparison visualizations.

    Shows what-if scenarios and minimal changes needed for different outcomes.
    """

    comparison_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    title: str = Field(default="Counterfactual Analysis")

    # Original state
    original_prediction: float = Field(..., description="Original prediction")
    original_class: Optional[str] = Field(default=None)
    original_instance: Dict[str, float] = Field(
        default_factory=dict,
        description="Original feature values"
    )

    # Counterfactual state
    counterfactual_prediction: float = Field(..., description="Counterfactual prediction")
    counterfactual_class: Optional[str] = Field(default=None)
    target_prediction: Optional[float] = Field(default=None, description="Target prediction")

    # Changes required
    changes: List[CounterfactualChange] = Field(
        default_factory=list,
        description="Required feature changes"
    )
    num_features_changed: int = Field(default=0, ge=0)
    total_change_magnitude: float = Field(default=0.0, ge=0)

    # Feasibility assessment
    overall_feasibility: float = Field(..., ge=0, le=1)
    feasibility_breakdown: Dict[str, float] = Field(default_factory=dict)

    # Human-readable explanation
    explanation_text: str = Field(default="")

    # Metadata
    model_id: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = Field(default="")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None,
            float: lambda v: round(v, 6) if isinstance(v, float) else v
        }
    )


class PredictionHistoryEntry(BaseModel):
    """Single prediction entry in history."""

    entry_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = Field(...)

    # Prediction data
    prediction: float = Field(...)
    prediction_class: Optional[str] = Field(default=None)
    confidence: float = Field(..., ge=0, le=1)

    # Input summary
    input_summary: Dict[str, float] = Field(
        default_factory=dict,
        description="Key feature values"
    )

    # Explanation summary
    top_features: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="Top contributing features"
    )
    explanation_confidence: float = Field(..., ge=0, le=1)

    # Status
    was_correct: Optional[bool] = Field(default=None)
    actual_outcome: Optional[float] = Field(default=None)

    # Provenance
    provenance_hash: str = Field(default="")


class PredictionHistoryView(BaseModel):
    """
    Data structure for prediction history with explanations.

    Shows historical predictions with their explanations for trend analysis.
    """

    history_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    title: str = Field(default="Prediction History")

    # Query parameters
    model_id: str = Field(...)
    time_range: TimeRange = Field(default=TimeRange.LAST_DAY)
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)

    # History entries
    entries: List[PredictionHistoryEntry] = Field(default_factory=list)
    total_count: int = Field(default=0, ge=0)

    # Summary statistics
    avg_prediction: float = Field(default=0.0)
    prediction_std: float = Field(default=0.0)
    min_prediction: float = Field(default=0.0)
    max_prediction: float = Field(default=0.0)
    avg_confidence: float = Field(default=0.0)

    # Trend data for charts
    trend_timestamps: List[str] = Field(default_factory=list)
    trend_predictions: List[float] = Field(default_factory=list)
    trend_confidences: List[float] = Field(default_factory=list)

    # Feature trend (how top feature changes over time)
    feature_trends: Dict[str, List[float]] = Field(default_factory=dict)

    # Pagination
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)
    total_pages: int = Field(default=1, ge=1)
    has_next: bool = Field(default=False)
    has_prev: bool = Field(default=False)

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None,
            float: lambda v: round(v, 6) if isinstance(v, float) else v
        }
    )


class ModelSummary(BaseModel):
    """Summary information for a single model."""

    model_id: str = Field(...)
    model_name: str = Field(...)
    model_type: str = Field(default="unknown")

    # Explanation stats
    total_explanations: int = Field(default=0, ge=0)
    avg_explanation_time_ms: float = Field(default=0.0, ge=0)
    last_explanation_time: Optional[datetime] = Field(default=None)

    # Feature importance
    top_features: List[str] = Field(default_factory=list)
    feature_count: int = Field(default=0, ge=0)

    # Quality metrics
    avg_confidence: float = Field(default=0.0, ge=0, le=1)
    data_quality_score: float = Field(default=0.0, ge=0, le=1)

    # Status
    is_active: bool = Field(default=True)
    has_training_data: bool = Field(default=False)


class ExplanationDashboardData(BaseModel):
    """
    Complete dashboard data for a model explanation.

    This is the main response model for the dashboard endpoint,
    containing all visualization data needed by the frontend.
    """

    dashboard_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    title: str = Field(default="ML Explainability Dashboard")

    # Model information
    model_id: str = Field(...)
    model_name: str = Field(default="Model")
    model_type: str = Field(default="unknown")

    # Current instance (if explaining specific prediction)
    instance_id: Optional[str] = Field(default=None)
    prediction: Optional[float] = Field(default=None)
    prediction_class: Optional[str] = Field(default=None)

    # Visualization components
    feature_contributions: Optional[FeatureContributionChart] = Field(default=None)
    global_importance: Optional[GlobalImportanceChart] = Field(default=None)
    counterfactual: Optional[CounterfactualComparisonView] = Field(default=None)
    history: Optional[PredictionHistoryView] = Field(default=None)

    # Human-readable explanation
    explanation_text: Optional[str] = Field(default=None)
    explanation_confidence: float = Field(default=0.0, ge=0, le=1)

    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_ms: float = Field(default=0.0, ge=0)
    provenance_hash: str = Field(default="")

    # View settings
    view_mode: DashboardViewMode = Field(default=DashboardViewMode.SUMMARY)

    # User preferences (if any)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None,
            float: lambda v: round(v, 6) if isinstance(v, float) else v
        }
    )


class DashboardSummary(BaseModel):
    """
    Overall explainability summary across all models.

    Provides a high-level overview of the explainability system status.
    """

    summary_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    title: str = Field(default="Explainability System Summary")

    # Model counts
    total_models: int = Field(default=0, ge=0)
    active_models: int = Field(default=0, ge=0)
    models_with_explanations: int = Field(default=0, ge=0)

    # Model summaries
    models: List[ModelSummary] = Field(default_factory=list)

    # System-wide statistics
    total_explanations_generated: int = Field(default=0, ge=0)
    explanations_last_24h: int = Field(default=0, ge=0)
    avg_explanation_time_ms: float = Field(default=0.0, ge=0)
    cache_hit_rate: float = Field(default=0.0, ge=0, le=1)

    # Health metrics
    system_health: str = Field(default="healthy")
    shap_available: bool = Field(default=True)
    lime_available: bool = Field(default=True)

    # Timestamp
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None,
            float: lambda v: round(v, 6) if isinstance(v, float) else v
        }
    )


# =============================================================================
# VISUALIZATION DATA GENERATORS
# =============================================================================

class VisualizationDataGenerator:
    """
    Generates visualization-ready data from raw explanation results.

    Transforms SHAP values, LIME weights, and other explanation outputs
    into structures suitable for frontend charting libraries (Plotly, Recharts).
    """

    # Default color schemes
    COLOR_POSITIVE = "#4caf50"  # Green
    COLOR_NEGATIVE = "#f44336"  # Red
    COLOR_NEUTRAL = "#9e9e9e"   # Gray
    COLOR_BASE = "#2196f3"      # Blue

    @staticmethod
    def generate_waterfall_data(
        feature_contributions: Dict[str, float],
        feature_values: Dict[str, float],
        base_value: float,
        prediction: float,
        model_id: str,
        feature_display_names: Optional[Dict[str, str]] = None,
        max_features: int = 10,
        instance_id: Optional[str] = None,
    ) -> FeatureContributionChart:
        """
        Generate SHAP waterfall chart data.

        The waterfall chart shows how each feature contribution
        adds or subtracts from the base value to reach the prediction.

        Args:
            feature_contributions: Dict of feature -> contribution value
            feature_values: Dict of feature -> actual value
            base_value: Model's expected/base value
            prediction: Final prediction
            model_id: Model identifier
            feature_display_names: Optional mapping to display names
            max_features: Maximum features to include
            instance_id: Optional instance identifier

        Returns:
            FeatureContributionChart ready for rendering
        """
        display_names = feature_display_names or {}

        # Sort by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:max_features]

        # Calculate total contribution for percentages
        total_abs_contribution = sum(abs(v) for _, v in sorted_features)
        if total_abs_contribution == 0:
            total_abs_contribution = 1e-10

        # Build contribution data
        contributions = []
        cumulative = [base_value]

        for rank, (feature, contribution) in enumerate(sorted_features, 1):
            value = feature_values.get(feature, 0.0)
            percentage = (abs(contribution) / total_abs_contribution) * 100
            direction = "positive" if contribution >= 0 else "negative"
            color = (
                VisualizationDataGenerator.COLOR_POSITIVE if contribution >= 0
                else VisualizationDataGenerator.COLOR_NEGATIVE
            )

            contributions.append(FeatureContributionData(
                feature_name=feature,
                display_name=display_names.get(feature, feature.replace("_", " ").title()),
                value=float(value),
                contribution=float(contribution),
                percentage=float(percentage),
                direction=direction,
                color=color,
                rank=rank,
            ))

            cumulative.append(cumulative[-1] + contribution)

        # Compute provenance hash
        provenance_data = {
            "model_id": model_id,
            "base_value": base_value,
            "prediction": prediction,
            "contributions": {f: c for f, c in sorted_features},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return FeatureContributionChart(
            chart_type=ChartType.WATERFALL,
            title="Feature Contributions (Waterfall)",
            subtitle=f"Base value: {base_value:.4f} | Prediction: {prediction:.4f}",
            base_value=float(base_value),
            prediction=float(prediction),
            contributions=contributions,
            cumulative_values=[float(v) for v in cumulative],
            model_id=model_id,
            instance_id=instance_id,
            provenance_hash=provenance_hash,
            max_features=max_features,
        )

    @staticmethod
    def generate_force_plot_data(
        feature_contributions: Dict[str, float],
        feature_values: Dict[str, float],
        base_value: float,
        prediction: float,
        model_id: str,
        feature_display_names: Optional[Dict[str, str]] = None,
        max_features: int = 15,
        instance_id: Optional[str] = None,
    ) -> FeatureContributionChart:
        """
        Generate SHAP force plot data.

        The force plot shows features pushing the prediction up or down
        from the base value, with feature magnitudes proportional to impact.

        Args:
            feature_contributions: Dict of feature -> contribution value
            feature_values: Dict of feature -> actual value
            base_value: Model's expected/base value
            prediction: Final prediction
            model_id: Model identifier
            feature_display_names: Optional mapping to display names
            max_features: Maximum features to include
            instance_id: Optional instance identifier

        Returns:
            FeatureContributionChart for force plot rendering
        """
        display_names = feature_display_names or {}

        # Separate positive and negative contributions
        positive_features = []
        negative_features = []

        for feature, contribution in feature_contributions.items():
            entry = (feature, contribution, feature_values.get(feature, 0.0))
            if contribution >= 0:
                positive_features.append(entry)
            else:
                negative_features.append(entry)

        # Sort each group by magnitude
        positive_features.sort(key=lambda x: x[1], reverse=True)
        negative_features.sort(key=lambda x: abs(x[1]), reverse=True)

        # Combine: positive pushing right, negative pushing left
        all_features = negative_features[:max_features // 2] + positive_features[:max_features // 2]

        # Calculate totals
        total_abs = sum(abs(c) for _, c, _ in all_features) or 1e-10

        contributions = []
        for rank, (feature, contribution, value) in enumerate(all_features, 1):
            percentage = (abs(contribution) / total_abs) * 100
            direction = "positive" if contribution >= 0 else "negative"

            contributions.append(FeatureContributionData(
                feature_name=feature,
                display_name=display_names.get(feature, feature.replace("_", " ").title()),
                value=float(value),
                contribution=float(contribution),
                percentage=float(percentage),
                direction=direction,
                color=(
                    VisualizationDataGenerator.COLOR_POSITIVE if contribution >= 0
                    else VisualizationDataGenerator.COLOR_NEGATIVE
                ),
                rank=rank,
            ))

        # Provenance
        provenance_hash = hashlib.sha256(
            json.dumps({
                "model_id": model_id,
                "prediction": prediction,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, sort_keys=True).encode()
        ).hexdigest()

        return FeatureContributionChart(
            chart_type=ChartType.FORCE_PLOT,
            title="Feature Contributions (Force Plot)",
            subtitle=f"Prediction: {prediction:.4f}",
            base_value=float(base_value),
            prediction=float(prediction),
            contributions=contributions,
            cumulative_values=[],
            model_id=model_id,
            instance_id=instance_id,
            provenance_hash=provenance_hash,
            max_features=max_features,
        )

    @staticmethod
    def generate_bar_chart_data(
        feature_importance: Dict[str, float],
        model_id: str,
        sample_size: int = 100,
        feature_display_names: Optional[Dict[str, str]] = None,
        importance_std: Optional[Dict[str, float]] = None,
        method: str = "shap",
        title: str = "Global Feature Importance",
    ) -> GlobalImportanceChart:
        """
        Generate global feature importance bar chart data.

        Args:
            feature_importance: Dict of feature -> importance score
            model_id: Model identifier
            sample_size: Number of samples used to compute importance
            feature_display_names: Optional mapping to display names
            importance_std: Optional standard deviations
            method: Method used (shap, permutation, etc.)
            title: Chart title

        Returns:
            GlobalImportanceChart ready for rendering
        """
        display_names = feature_display_names or {}
        std_values = importance_std or {}

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        features = [f for f, _ in sorted_features]
        display_list = [display_names.get(f, f.replace("_", " ").title()) for f in features]
        scores = [float(s) for _, s in sorted_features]
        stds = [float(std_values.get(f, 0.0)) for f in features] if std_values else None

        top_feature = features[0] if features else ""
        top_importance = scores[0] if scores else 0.0

        # Provenance
        provenance_hash = hashlib.sha256(
            json.dumps({
                "model_id": model_id,
                "importance": dict(sorted_features),
                "method": method,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, sort_keys=True).encode()
        ).hexdigest()

        return GlobalImportanceChart(
            chart_type=ChartType.BAR_CHART,
            title=title,
            subtitle=f"Computed using {method.upper()} on {sample_size} samples",
            features=features,
            display_names=display_list,
            importance_scores=scores,
            importance_std=stds,
            total_features=len(features),
            sample_size=sample_size,
            top_feature=top_feature,
            top_importance=top_importance,
            model_id=model_id,
            method=method,
            provenance_hash=provenance_hash,
        )

    @staticmethod
    def generate_counterfactual_view(
        original_instance: Dict[str, float],
        original_prediction: float,
        counterfactual_instance: Dict[str, float],
        counterfactual_prediction: float,
        model_id: str,
        target_prediction: Optional[float] = None,
        feature_display_names: Optional[Dict[str, str]] = None,
        feature_units: Optional[Dict[str, str]] = None,
        explanation_text: Optional[str] = None,
    ) -> CounterfactualComparisonView:
        """
        Generate counterfactual comparison view data.

        Args:
            original_instance: Original feature values
            original_prediction: Original prediction
            counterfactual_instance: Counterfactual feature values
            counterfactual_prediction: Counterfactual prediction
            model_id: Model identifier
            target_prediction: Optional target prediction
            feature_display_names: Optional mapping to display names
            feature_units: Optional feature units
            explanation_text: Optional human-readable explanation

        Returns:
            CounterfactualComparisonView ready for rendering
        """
        display_names = feature_display_names or {}
        units = feature_units or {}

        # Calculate changes
        changes = []
        feasibility_scores = []

        for feature, original_value in original_instance.items():
            cf_value = counterfactual_instance.get(feature, original_value)

            if abs(cf_value - original_value) > 1e-6:
                change_amount = cf_value - original_value
                change_pct = (abs(change_amount) / (abs(original_value) + 1e-10)) * 100
                direction = "increase" if change_amount > 0 else "decrease"

                # Simple feasibility based on change magnitude
                feasibility = max(0.0, 1.0 - min(change_pct / 100, 1.0))
                feasibility_scores.append(feasibility)

                changes.append(CounterfactualChange(
                    feature_name=feature,
                    display_name=display_names.get(feature, feature.replace("_", " ").title()),
                    original_value=float(original_value),
                    counterfactual_value=float(cf_value),
                    change_amount=float(abs(change_amount)),
                    change_percentage=float(change_pct),
                    direction=direction,
                    feasibility=float(feasibility),
                    units=units.get(feature),
                ))

        # Sort changes by magnitude
        changes.sort(key=lambda x: x.change_amount, reverse=True)

        # Overall feasibility
        overall_feasibility = sum(feasibility_scores) / len(feasibility_scores) if feasibility_scores else 1.0

        # Total change magnitude
        total_magnitude = sum(c.change_amount for c in changes)

        # Generate explanation if not provided
        if not explanation_text and changes:
            lines = [
                f"To change the prediction from {original_prediction:.2%} to {counterfactual_prediction:.2%}:"
            ]
            for i, change in enumerate(changes[:5], 1):
                lines.append(
                    f"  {i}. {change.direction.capitalize()} {change.display_name} "
                    f"from {change.original_value:.2f} to {change.counterfactual_value:.2f}"
                )
            explanation_text = "\n".join(lines)

        # Provenance
        provenance_hash = hashlib.sha256(
            json.dumps({
                "model_id": model_id,
                "original_prediction": original_prediction,
                "counterfactual_prediction": counterfactual_prediction,
                "changes": [(c.feature_name, c.change_amount) for c in changes],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, sort_keys=True).encode()
        ).hexdigest()

        return CounterfactualComparisonView(
            title="Counterfactual Analysis",
            original_prediction=float(original_prediction),
            original_instance=original_instance,
            counterfactual_prediction=float(counterfactual_prediction),
            target_prediction=float(target_prediction) if target_prediction else None,
            changes=changes,
            num_features_changed=len(changes),
            total_change_magnitude=float(total_magnitude),
            overall_feasibility=float(overall_feasibility),
            feasibility_breakdown={c.feature_name: c.feasibility for c in changes},
            explanation_text=explanation_text or "",
            model_id=model_id,
            provenance_hash=provenance_hash,
        )

    @staticmethod
    def generate_time_series_data(
        predictions: List[Tuple[datetime, float]],
        explanations: List[Tuple[datetime, Dict[str, float]]],
        model_id: str,
        time_range: TimeRange = TimeRange.LAST_DAY,
    ) -> PredictionHistoryView:
        """
        Generate time series explanation data.

        Args:
            predictions: List of (timestamp, prediction) tuples
            explanations: List of (timestamp, feature_contributions) tuples
            model_id: Model identifier
            time_range: Time range for the history

        Returns:
            PredictionHistoryView ready for rendering
        """
        if not predictions:
            return PredictionHistoryView(
                model_id=model_id,
                time_range=time_range,
                entries=[],
                total_count=0,
            )

        entries = []
        all_predictions = []
        all_confidences = []
        trend_timestamps = []
        trend_predictions = []
        trend_confidences = []
        feature_values_by_time: Dict[str, List[float]] = {}

        for i, (ts, pred) in enumerate(predictions):
            all_predictions.append(pred)

            # Get explanation for this timestamp
            exp = explanations[i][1] if i < len(explanations) else {}

            # Top features
            sorted_features = sorted(exp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

            # Confidence based on explanation coverage
            total_contrib = sum(abs(v) for v in exp.values()) if exp else 0
            confidence = min(0.95, 0.5 + 0.45 * (total_contrib / (abs(pred) + 1e-10)))
            all_confidences.append(confidence)

            entries.append(PredictionHistoryEntry(
                timestamp=ts,
                prediction=float(pred),
                confidence=float(confidence),
                input_summary={f: exp.get(f, 0) for f, _ in sorted_features[:3]},
                top_features=[(f, v) for f, v in sorted_features],
                explanation_confidence=float(confidence),
                provenance_hash=hashlib.sha256(
                    f"{model_id}:{ts.isoformat()}:{pred}".encode()
                ).hexdigest()[:16],
            ))

            # Trend data
            trend_timestamps.append(ts.isoformat())
            trend_predictions.append(float(pred))
            trend_confidences.append(float(confidence))

            # Track feature values over time
            for f, v in exp.items():
                if f not in feature_values_by_time:
                    feature_values_by_time[f] = []
                feature_values_by_time[f].append(v)

        # Summary statistics
        pred_array = np.array(all_predictions)

        return PredictionHistoryView(
            model_id=model_id,
            time_range=time_range,
            start_time=predictions[0][0] if predictions else None,
            end_time=predictions[-1][0] if predictions else None,
            entries=entries,
            total_count=len(entries),
            avg_prediction=float(np.mean(pred_array)),
            prediction_std=float(np.std(pred_array)),
            min_prediction=float(np.min(pred_array)),
            max_prediction=float(np.max(pred_array)),
            avg_confidence=float(np.mean(all_confidences)) if all_confidences else 0.0,
            trend_timestamps=trend_timestamps,
            trend_predictions=trend_predictions,
            trend_confidences=trend_confidences,
            feature_trends=feature_values_by_time,
            page=1,
            page_size=len(entries),
            total_pages=1,
        )


# =============================================================================
# DASHBOARD STATE MANAGEMENT
# =============================================================================

class DashboardStateManager:
    """
    Manages dashboard state including model tracking, caching, and user preferences.

    Provides:
    - Selected model tracking
    - Explanation cache management with LRU eviction
    - User preference storage
    - Export functionality
    """

    def __init__(
        self,
        cache_size: int = 1000,
        cache_ttl_seconds: float = 3600.0,
    ):
        """
        Initialize dashboard state manager.

        Args:
            cache_size: Maximum cache entries
            cache_ttl_seconds: Cache entry time-to-live
        """
        self._selected_model: Optional[str] = None
        self._model_registry: Dict[str, Any] = {}
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl_seconds
        self._user_preferences: Dict[str, Dict[str, Any]] = {}
        self._explanation_history: Dict[str, List[ExplanationDashboardData]] = {}

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_explanations = 0

        logger.info(
            f"DashboardStateManager initialized: cache_size={cache_size}, ttl={cache_ttl_seconds}s"
        )

    def select_model(self, model_id: str) -> None:
        """Set the currently selected model."""
        self._selected_model = model_id
        logger.debug(f"Selected model: {model_id}")

    def get_selected_model(self) -> Optional[str]:
        """Get the currently selected model ID."""
        return self._selected_model

    def register_model(
        self,
        model_id: str,
        model: Any,
        feature_names: List[str],
        training_data: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a model for explanation.

        Args:
            model_id: Unique model identifier
            model: ML model with predict method
            feature_names: List of feature names
            training_data: Optional training data for SHAP
            metadata: Optional model metadata
        """
        self._model_registry[model_id] = {
            "model": model,
            "feature_names": feature_names,
            "training_data": training_data,
            "metadata": metadata or {},
            "registered_at": datetime.now(timezone.utc),
        }
        logger.info(f"Model registered: {model_id}")

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get registered model information."""
        return self._model_registry.get(model_id)

    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self._model_registry.keys())

    def _generate_cache_key(self, model_id: str, instance_hash: str, method: str) -> str:
        """Generate cache key from model, instance, and method."""
        return f"{model_id}:{instance_hash}:{method}"

    def get_cached_explanation(
        self,
        model_id: str,
        instance_hash: str,
        method: str = "shap",
    ) -> Optional[Any]:
        """
        Get cached explanation if available and not expired.

        Args:
            model_id: Model identifier
            instance_hash: Hash of the instance data
            method: Explanation method

        Returns:
            Cached explanation or None
        """
        key = self._generate_cache_key(model_id, instance_hash, method)

        if key in self._cache:
            entry, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp < self._cache_ttl:
                self._cache_hits += 1
                # Move to end (LRU)
                self._cache.move_to_end(key)
                return entry
            else:
                # Expired
                del self._cache[key]

        self._cache_misses += 1
        return None

    def cache_explanation(
        self,
        model_id: str,
        instance_hash: str,
        explanation: Any,
        method: str = "shap",
    ) -> None:
        """
        Cache an explanation result.

        Args:
            model_id: Model identifier
            instance_hash: Hash of the instance data
            explanation: Explanation result to cache
            method: Explanation method
        """
        key = self._generate_cache_key(model_id, instance_hash, method)

        # Evict oldest if at capacity
        while len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)

        self._cache[key] = (explanation, time.time())

    def clear_cache(self, model_id: Optional[str] = None) -> int:
        """
        Clear the explanation cache.

        Args:
            model_id: Optional model to clear cache for (all if None)

        Returns:
            Number of entries cleared
        """
        if model_id is None:
            count = len(self._cache)
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            return count
        else:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{model_id}:")]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)

    @property
    def cache_hit_rate(self) -> float:
        """Get current cache hit rate."""
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0

    def set_user_preference(
        self,
        user_id: str,
        preference_key: str,
        value: Any,
    ) -> None:
        """
        Set a user preference.

        Args:
            user_id: User identifier
            preference_key: Preference key
            value: Preference value
        """
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}
        self._user_preferences[user_id][preference_key] = value

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all preferences for a user."""
        return self._user_preferences.get(user_id, {})

    def add_to_history(
        self,
        model_id: str,
        dashboard_data: ExplanationDashboardData,
        max_history: int = 1000,
    ) -> None:
        """
        Add an explanation to history.

        Args:
            model_id: Model identifier
            dashboard_data: Dashboard data to record
            max_history: Maximum history entries per model
        """
        if model_id not in self._explanation_history:
            self._explanation_history[model_id] = []

        self._explanation_history[model_id].append(dashboard_data)
        self._total_explanations += 1

        # Trim if over limit
        while len(self._explanation_history[model_id]) > max_history:
            self._explanation_history[model_id].pop(0)

    def get_history(
        self,
        model_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ExplanationDashboardData]:
        """Get explanation history for a model."""
        history = self._explanation_history.get(model_id, [])
        return history[offset:offset + limit]

    def export_data(
        self,
        data: Union[ExplanationDashboardData, GlobalImportanceChart, FeatureContributionChart],
        format: ExportFormat = ExportFormat.JSON,
    ) -> Union[str, bytes]:
        """
        Export dashboard data in specified format.

        Args:
            data: Data to export
            format: Export format

        Returns:
            Exported data as string or bytes
        """
        if format == ExportFormat.JSON:
            return data.model_dump_json(indent=2)

        elif format == ExportFormat.CSV:
            # Convert to CSV format
            lines = []

            if isinstance(data, FeatureContributionChart):
                lines.append("feature,display_name,value,contribution,percentage,direction")
                for c in data.contributions:
                    lines.append(
                        f"{c.feature_name},{c.display_name},{c.value},"
                        f"{c.contribution},{c.percentage},{c.direction}"
                    )

            elif isinstance(data, GlobalImportanceChart):
                lines.append("feature,display_name,importance")
                for i, f in enumerate(data.features):
                    lines.append(
                        f"{f},{data.display_names[i]},{data.importance_scores[i]}"
                    )

            elif isinstance(data, ExplanationDashboardData):
                lines.append("field,value")
                lines.append(f"model_id,{data.model_id}")
                lines.append(f"prediction,{data.prediction}")
                lines.append(f"timestamp,{data.timestamp}")
                lines.append(f"provenance_hash,{data.provenance_hash}")

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_summary(self) -> DashboardSummary:
        """
        Get overall dashboard summary.

        Returns:
            DashboardSummary with system-wide statistics
        """
        models = []

        for model_id, info in self._model_registry.items():
            history = self._explanation_history.get(model_id, [])

            # Calculate stats from history
            if history:
                avg_time = sum(
                    h.processing_time_ms for h in history
                ) / len(history)
                avg_conf = sum(
                    h.explanation_confidence for h in history
                ) / len(history)
                last_time = history[-1].timestamp
            else:
                avg_time = 0.0
                avg_conf = 0.0
                last_time = None

            models.append(ModelSummary(
                model_id=model_id,
                model_name=info["metadata"].get("name", model_id),
                model_type=info["metadata"].get("type", "unknown"),
                total_explanations=len(history),
                avg_explanation_time_ms=avg_time,
                last_explanation_time=last_time,
                feature_count=len(info["feature_names"]),
                avg_confidence=avg_conf,
                has_training_data=info["training_data"] is not None,
            ))

        # Check library availability
        shap_available = False
        lime_available = False
        try:
            import shap
            shap_available = True
        except ImportError:
            pass
        try:
            import lime
            lime_available = True
        except ImportError:
            pass

        return DashboardSummary(
            total_models=len(self._model_registry),
            active_models=len(self._model_registry),
            models_with_explanations=len(self._explanation_history),
            models=models,
            total_explanations_generated=self._total_explanations,
            avg_explanation_time_ms=sum(
                m.avg_explanation_time_ms for m in models
            ) / len(models) if models else 0.0,
            cache_hit_rate=self.cache_hit_rate,
            shap_available=shap_available,
            lime_available=lime_available,
        )


# =============================================================================
# GLOBAL STATE INSTANCE
# =============================================================================

# Global dashboard state manager instance
dashboard_state = DashboardStateManager()


def get_dashboard_state() -> DashboardStateManager:
    """Dependency injection for dashboard state."""
    return dashboard_state


# =============================================================================
# FASTAPI ROUTER
# =============================================================================

if FASTAPI_AVAILABLE:
    dashboard_router = APIRouter(
        prefix="/dashboard",
        tags=["Explainability Dashboard"],
    )

    @dashboard_router.get(
        "/explanation/{model_id}",
        response_model=ExplanationDashboardData,
        summary="Get explanation visualization data",
        description="Returns complete dashboard data for visualizing model explanations"
    )
    async def get_explanation_dashboard(
        model_id: str,
        instance_data: Optional[str] = Query(
            None,
            description="JSON-encoded instance data for explanation"
        ),
        include_global: bool = Query(
            True,
            description="Include global importance"
        ),
        include_counterfactual: bool = Query(
            False,
            description="Include counterfactual analysis"
        ),
        max_features: int = Query(
            10,
            ge=1,
            le=50,
            description="Maximum features to display"
        ),
        state: DashboardStateManager = Depends(get_dashboard_state),
    ) -> ExplanationDashboardData:
        """
        Get explanation visualization data for a model.

        Returns comprehensive dashboard data including:
        - Feature contributions (waterfall/force plot)
        - Global feature importance
        - Optional counterfactual analysis
        """
        start_time = time.time()

        # Get model from registry
        model_info = state.get_model(model_id)
        if model_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_id}"
            )

        model = model_info["model"]
        feature_names = model_info["feature_names"]
        training_data = model_info["training_data"]

        # Parse instance data if provided
        instance = None
        instance_hash = ""
        prediction = None

        if instance_data:
            try:
                instance = json.loads(instance_data)
                instance_hash = hashlib.md5(instance_data.encode()).hexdigest()[:16]
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON for instance_data"
                )

        # Check cache
        cached = state.get_cached_explanation(model_id, instance_hash, "shap")
        if cached is not None:
            return cached

        # Generate visualizations
        feature_contributions = None
        global_importance = None
        counterfactual = None
        explanation_text = None
        explanation_confidence = 0.0

        # Try to generate SHAP explanation for instance
        if instance:
            try:
                # Get prediction
                X = np.array([[instance.get(f, 0.0) for f in feature_names]])

                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)
                    prediction = float(pred[0, 1]) if len(pred.shape) > 1 and pred.shape[1] > 1 else float(pred[0])
                else:
                    prediction = float(model.predict(X)[0])

                # Try SHAP explanation
                try:
                    import shap

                    # Create explainer
                    if training_data is not None and len(training_data) > 10:
                        background = shap.kmeans(training_data, min(100, len(training_data)))
                        if hasattr(model, "predict_proba"):
                            explainer = shap.KernelExplainer(
                                lambda x: model.predict_proba(x)[:, 1], background
                            )
                        else:
                            explainer = shap.KernelExplainer(model.predict, background)
                    else:
                        try:
                            explainer = shap.TreeExplainer(model)
                        except Exception:
                            if hasattr(model, "predict_proba"):
                                explainer = shap.KernelExplainer(
                                    lambda x: model.predict_proba(x)[:, 1], X
                                )
                            else:
                                explainer = shap.KernelExplainer(model.predict, X)

                    # Get SHAP values
                    shap_values = explainer.shap_values(X)

                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

                    # Build contribution dict
                    contributions = {
                        name: float(shap_values[0, i])
                        for i, name in enumerate(feature_names)
                    }

                    base_value = float(
                        explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray))
                        and len(explainer.expected_value) > 1
                        else explainer.expected_value
                    ) if hasattr(explainer, "expected_value") else 0.5

                    # Generate waterfall chart
                    feature_contributions = VisualizationDataGenerator.generate_waterfall_data(
                        feature_contributions=contributions,
                        feature_values=instance,
                        base_value=base_value,
                        prediction=prediction,
                        model_id=model_id,
                        max_features=max_features,
                        instance_id=instance_hash,
                    )

                    # Calculate confidence
                    total_contrib = sum(abs(v) for v in contributions.values())
                    explanation_confidence = min(0.95, 0.5 + 0.45 * min(total_contrib, 1.0))

                    # Generate explanation text
                    top_features = sorted(
                        contributions.items(), key=lambda x: abs(x[1]), reverse=True
                    )[:3]
                    explanation_text = f"Prediction: {prediction:.2%}. "
                    explanation_text += "Key factors: "
                    explanation_text += ", ".join([
                        f"{f} ({'increases' if v > 0 else 'decreases'} by {abs(v):.2%})"
                        for f, v in top_features
                    ])

                except ImportError:
                    logger.warning("SHAP not available for instance explanation")
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")

            except Exception as e:
                logger.error(f"Instance explanation failed: {e}")

        # Generate global importance
        if include_global and training_data is not None:
            try:
                import shap

                # Use subset for speed
                sample_size = min(100, len(training_data))
                sample_idx = np.random.choice(len(training_data), sample_size, replace=False)
                X_sample = training_data[sample_idx]

                try:
                    explainer = shap.TreeExplainer(model)
                except Exception:
                    background = shap.kmeans(X_sample, min(50, len(X_sample)))
                    if hasattr(model, "predict_proba"):
                        explainer = shap.KernelExplainer(
                            lambda x: model.predict_proba(x)[:, 1], background
                        )
                    else:
                        explainer = shap.KernelExplainer(model.predict, background)

                shap_values = explainer.shap_values(X_sample)

                if isinstance(shap_values, list):
                    shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                else:
                    shap_values = np.abs(shap_values)

                mean_importance = shap_values.mean(axis=0)
                total = mean_importance.sum() + 1e-10

                importance = {
                    name: float(mean_importance[i] / total)
                    for i, name in enumerate(feature_names)
                }

                global_importance = VisualizationDataGenerator.generate_bar_chart_data(
                    feature_importance=importance,
                    model_id=model_id,
                    sample_size=sample_size,
                )

            except ImportError:
                logger.warning("SHAP not available for global importance")
            except Exception as e:
                logger.warning(f"Global importance computation failed: {e}")

        # Generate counterfactual (if requested and instance provided)
        if include_counterfactual and instance and prediction is not None:
            try:
                from .counterfactual import CounterfactualExplainer

                cf_explainer = CounterfactualExplainer(
                    model=model,
                    feature_names=feature_names,
                )

                # Target: flip prediction
                target = 0.2 if prediction > 0.5 else 0.8

                cf_result = cf_explainer.generate_counterfactual(
                    instance=instance,
                    target_prediction=target,
                )

                # Build counterfactual instance
                cf_instance = instance.copy()
                for feature, (old, new) in cf_result.changes_required.items():
                    cf_instance[feature] = new

                counterfactual = VisualizationDataGenerator.generate_counterfactual_view(
                    original_instance=instance,
                    original_prediction=prediction,
                    counterfactual_instance=cf_instance,
                    counterfactual_prediction=cf_result.counterfactual_prediction,
                    model_id=model_id,
                    target_prediction=target,
                    explanation_text=cf_result.explanation_text,
                )

            except Exception as e:
                logger.warning(f"Counterfactual generation failed: {e}")

        processing_time = (time.time() - start_time) * 1000

        # Compute dashboard provenance
        provenance_hash = hashlib.sha256(
            json.dumps({
                "model_id": model_id,
                "instance_hash": instance_hash,
                "prediction": prediction,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, sort_keys=True).encode()
        ).hexdigest()

        dashboard_data = ExplanationDashboardData(
            model_id=model_id,
            model_name=model_info["metadata"].get("name", model_id),
            model_type=model_info["metadata"].get("type", "unknown"),
            instance_id=instance_hash if instance else None,
            prediction=prediction,
            feature_contributions=feature_contributions,
            global_importance=global_importance,
            counterfactual=counterfactual,
            explanation_text=explanation_text,
            explanation_confidence=explanation_confidence,
            processing_time_ms=processing_time,
            provenance_hash=provenance_hash,
        )

        # Cache and record
        if instance_hash:
            state.cache_explanation(model_id, instance_hash, dashboard_data)
        state.add_to_history(model_id, dashboard_data)

        return dashboard_data

    @dashboard_router.get(
        "/feature-importance/{model_id}",
        response_model=GlobalImportanceChart,
        summary="Get feature importance chart data",
        description="Returns global feature importance data for visualization"
    )
    async def get_feature_importance(
        model_id: str,
        method: str = Query(
            "shap",
            description="Importance method: shap, permutation"
        ),
        sample_size: int = Query(
            100,
            ge=10,
            le=1000,
            description="Number of samples for computation"
        ),
        state: DashboardStateManager = Depends(get_dashboard_state),
    ) -> GlobalImportanceChart:
        """
        Get global feature importance for a model.

        Computes model-wide feature importance using SHAP or permutation.
        """
        # Get model
        model_info = state.get_model(model_id)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        model = model_info["model"]
        feature_names = model_info["feature_names"]
        training_data = model_info["training_data"]

        if training_data is None:
            raise HTTPException(
                status_code=400,
                detail="Training data required for feature importance"
            )

        try:
            import shap

            # Sample data
            actual_sample_size = min(sample_size, len(training_data))
            sample_idx = np.random.choice(len(training_data), actual_sample_size, replace=False)
            X_sample = training_data[sample_idx]

            # Create explainer
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                background = shap.kmeans(X_sample, min(50, len(X_sample)))
                if hasattr(model, "predict_proba"):
                    explainer = shap.KernelExplainer(
                        lambda x: model.predict_proba(x)[:, 1], background
                    )
                else:
                    explainer = shap.KernelExplainer(model.predict, background)

            # Compute SHAP values
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values = np.abs(shap_values)

            mean_importance = shap_values.mean(axis=0)
            std_importance = shap_values.std(axis=0)
            total = mean_importance.sum() + 1e-10

            importance = {
                name: float(mean_importance[i] / total)
                for i, name in enumerate(feature_names)
            }

            importance_std = {
                name: float(std_importance[i] / total)
                for i, name in enumerate(feature_names)
            }

            return VisualizationDataGenerator.generate_bar_chart_data(
                feature_importance=importance,
                model_id=model_id,
                sample_size=actual_sample_size,
                importance_std=importance_std,
                method=method,
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="SHAP library not available"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Feature importance computation failed: {str(e)}"
            )

    @dashboard_router.get(
        "/counterfactuals/{model_id}",
        response_model=CounterfactualComparisonView,
        summary="Get counterfactual comparison data",
        description="Returns counterfactual analysis for an instance"
    )
    async def get_counterfactuals(
        model_id: str,
        instance_data: str = Query(
            ...,
            description="JSON-encoded instance data"
        ),
        target_prediction: Optional[float] = Query(
            None,
            description="Target prediction to achieve"
        ),
        max_features_to_change: int = Query(
            5,
            ge=1,
            le=20,
            description="Maximum features to modify"
        ),
        state: DashboardStateManager = Depends(get_dashboard_state),
    ) -> CounterfactualComparisonView:
        """
        Get counterfactual explanation for an instance.

        Shows what minimal changes would be needed to achieve a different prediction.
        """
        # Get model
        model_info = state.get_model(model_id)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        model = model_info["model"]
        feature_names = model_info["feature_names"]

        # Parse instance
        try:
            instance = json.loads(instance_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON for instance_data")

        try:
            from .counterfactual import CounterfactualExplainer

            # Get original prediction
            X = np.array([[instance.get(f, 0.0) for f in feature_names]])
            if hasattr(model, "predict_proba"):
                original_pred = float(model.predict_proba(X)[0, 1])
            else:
                original_pred = float(model.predict(X)[0])

            # Determine target
            if target_prediction is None:
                target_prediction = 0.2 if original_pred > 0.5 else 0.8

            # Generate counterfactual
            explainer = CounterfactualExplainer(
                model=model,
                feature_names=feature_names,
            )

            result = explainer.generate_counterfactual(
                instance=instance,
                target_prediction=target_prediction,
                max_features_to_change=max_features_to_change,
            )

            # Build counterfactual instance
            cf_instance = instance.copy()
            for feature, (old, new) in result.changes_required.items():
                cf_instance[feature] = new

            return VisualizationDataGenerator.generate_counterfactual_view(
                original_instance=instance,
                original_prediction=original_pred,
                counterfactual_instance=cf_instance,
                counterfactual_prediction=result.counterfactual_prediction,
                model_id=model_id,
                target_prediction=target_prediction,
                explanation_text=result.explanation_text,
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Counterfactual explainer not available"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Counterfactual generation failed: {str(e)}"
            )

    @dashboard_router.get(
        "/history/{model_id}",
        response_model=PredictionHistoryView,
        summary="Get prediction history with explanations",
        description="Returns historical predictions with their explanations"
    )
    async def get_prediction_history(
        model_id: str,
        time_range: TimeRange = Query(
            TimeRange.LAST_DAY,
            description="Time range for history"
        ),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(50, ge=1, le=500, description="Items per page"),
        state: DashboardStateManager = Depends(get_dashboard_state),
    ) -> PredictionHistoryView:
        """
        Get prediction history with explanations.

        Returns historical predictions and their explanations for trend analysis.
        """
        # Get model
        model_info = state.get_model(model_id)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        # Get history
        history = state.get_history(model_id, limit=page_size * page)

        if not history:
            return PredictionHistoryView(
                model_id=model_id,
                time_range=time_range,
                entries=[],
                total_count=0,
                page=page,
                page_size=page_size,
                total_pages=0,
            )

        # Convert to history entries
        entries = []
        predictions = []
        confidences = []
        timestamps = []

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_history = history[start_idx:end_idx]

        for h in page_history:
            if h.prediction is not None:
                entries.append(PredictionHistoryEntry(
                    timestamp=h.timestamp,
                    prediction=h.prediction,
                    confidence=h.explanation_confidence,
                    top_features=[(f.feature_name, f.contribution)
                                  for f in (h.feature_contributions.contributions[:3]
                                           if h.feature_contributions else [])],
                    explanation_confidence=h.explanation_confidence,
                    provenance_hash=h.provenance_hash[:16],
                ))
                predictions.append(h.prediction)
                confidences.append(h.explanation_confidence)
                timestamps.append(h.timestamp.isoformat())

        total_count = len(history)
        total_pages = (total_count + page_size - 1) // page_size

        return PredictionHistoryView(
            model_id=model_id,
            time_range=time_range,
            entries=entries,
            total_count=total_count,
            avg_prediction=float(np.mean(predictions)) if predictions else 0.0,
            prediction_std=float(np.std(predictions)) if predictions else 0.0,
            min_prediction=float(np.min(predictions)) if predictions else 0.0,
            max_prediction=float(np.max(predictions)) if predictions else 0.0,
            avg_confidence=float(np.mean(confidences)) if confidences else 0.0,
            trend_timestamps=timestamps,
            trend_predictions=predictions,
            trend_confidences=confidences,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )

    @dashboard_router.get(
        "/summary",
        response_model=DashboardSummary,
        summary="Get overall explainability summary",
        description="Returns system-wide explainability statistics"
    )
    async def get_dashboard_summary(
        state: DashboardStateManager = Depends(get_dashboard_state),
    ) -> DashboardSummary:
        """
        Get overall explainability system summary.

        Returns high-level overview of all models and explanation statistics.
        """
        return state.get_summary()

    @dashboard_router.post(
        "/export/{model_id}",
        summary="Export dashboard data",
        description="Export explanation data in various formats"
    )
    async def export_dashboard_data(
        model_id: str,
        format: ExportFormat = Query(
            ExportFormat.JSON,
            description="Export format"
        ),
        instance_data: Optional[str] = Query(
            None,
            description="JSON-encoded instance data"
        ),
        state: DashboardStateManager = Depends(get_dashboard_state),
    ):
        """
        Export dashboard data in specified format.

        Supports JSON and CSV export formats.
        """
        # Get model
        model_info = state.get_model(model_id)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        # Generate dashboard data
        dashboard_data = await get_explanation_dashboard(
            model_id=model_id,
            instance_data=instance_data,
            include_global=True,
            include_counterfactual=False,
            max_features=20,
            state=state,
        )

        # Export
        exported = state.export_data(dashboard_data, format)

        if format == ExportFormat.JSON:
            return JSONResponse(
                content=json.loads(exported),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=explanation_{model_id}.json"
                }
            )
        elif format == ExportFormat.CSV:
            return Response(
                content=exported,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=explanation_{model_id}.csv"
                }
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    @dashboard_router.post(
        "/models/register",
        summary="Register a model for explanation",
        description="Register a model in the dashboard state"
    )
    async def register_model(
        model_id: str = Query(..., description="Unique model identifier"),
        feature_names: str = Query(..., description="JSON-encoded feature names"),
        model_name: str = Query("Model", description="Human-readable model name"),
        model_type: str = Query("unknown", description="Model type"),
        state: DashboardStateManager = Depends(get_dashboard_state),
    ):
        """
        Register a model for dashboard tracking.

        Note: For demo purposes, creates a mock model. In production,
        models would be loaded from a model registry.
        """
        try:
            names = json.loads(feature_names)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON for feature_names")

        # Create mock model for demo
        class MockModel:
            def predict_proba(self, X):
                probs = 1 / (1 + np.exp(-np.sum(X, axis=1) / len(X[0])))
                return np.column_stack([1 - probs, probs])

            def predict(self, X):
                return np.sum(X, axis=1)

        # Generate synthetic training data
        training_data = np.random.randn(1000, len(names))

        state.register_model(
            model_id=model_id,
            model=MockModel(),
            feature_names=names,
            training_data=training_data,
            metadata={"name": model_name, "type": model_type},
        )

        return {"status": "success", "model_id": model_id}

    @dashboard_router.get(
        "/models",
        summary="List registered models",
        description="Returns list of models registered for explanation"
    )
    async def list_models(
        state: DashboardStateManager = Depends(get_dashboard_state),
    ):
        """List all registered models."""
        models = []
        for model_id in state.list_models():
            info = state.get_model(model_id)
            models.append({
                "model_id": model_id,
                "name": info["metadata"].get("name", model_id),
                "type": info["metadata"].get("type", "unknown"),
                "feature_count": len(info["feature_names"]),
                "has_training_data": info["training_data"] is not None,
            })
        return {"models": models}

    @dashboard_router.delete(
        "/cache",
        summary="Clear explanation cache",
        description="Clear cached explanations"
    )
    async def clear_cache(
        model_id: Optional[str] = Query(
            None,
            description="Model ID (all if not specified)"
        ),
        state: DashboardStateManager = Depends(get_dashboard_state),
    ):
        """Clear the explanation cache."""
        count = state.clear_cache(model_id)
        return {"status": "success", "cleared_entries": count}

else:
    # Stub router if FastAPI not available
    dashboard_router = None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ChartType",
    "ExportFormat",
    "TimeRange",
    "DashboardViewMode",
    # Data Models
    "FeatureContributionData",
    "FeatureContributionChart",
    "GlobalImportanceChart",
    "CounterfactualChange",
    "CounterfactualComparisonView",
    "PredictionHistoryEntry",
    "PredictionHistoryView",
    "ModelSummary",
    "ExplanationDashboardData",
    "DashboardSummary",
    # Visualization Generator
    "VisualizationDataGenerator",
    # State Management
    "DashboardStateManager",
    "dashboard_state",
    "get_dashboard_state",
    # Router
    "dashboard_router",
]


if __name__ == "__main__":
    # Example usage when run directly
    print("=" * 70)
    print("GreenLang ML Explainability Dashboard Module")
    print("=" * 70)
    print()
    print("Available Data Models:")
    for model in [
        "ExplanationDashboardData",
        "FeatureContributionChart",
        "GlobalImportanceChart",
        "CounterfactualComparisonView",
        "PredictionHistoryView",
        "DashboardSummary",
    ]:
        print(f"  - {model}")
    print()
    print("API Endpoints (when included in FastAPI app):")
    print("  GET  /dashboard/explanation/{model_id}")
    print("  GET  /dashboard/feature-importance/{model_id}")
    print("  GET  /dashboard/counterfactuals/{model_id}")
    print("  GET  /dashboard/history/{model_id}")
    print("  GET  /dashboard/summary")
    print("  POST /dashboard/export/{model_id}")
    print("  POST /dashboard/models/register")
    print("  GET  /dashboard/models")
    print("  DELETE /dashboard/cache")
    print()
    print("=" * 70)
