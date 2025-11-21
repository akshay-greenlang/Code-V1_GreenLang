# -*- coding: utf-8 -*-
"""
Experiment Data Models for A/B Testing Framework

This module defines Pydantic models for A/B testing experiments, variants,
results, and statistical analysis.

Example:
    >>> experiment = Experiment(
    ...     name="combustion_algorithm_test",
    ...     variants=[
    ...         ExperimentVariant(name="control", traffic_split=0.5, config={}),
    ...         ExperimentVariant(name="treatment", traffic_split=0.5, config={})
    ...     ]
    ... )
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from enum import Enum
import hashlib
from greenlang.determinism import DeterministicClock


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class MetricType(str, Enum):
    """Types of metrics tracked in experiments."""
    CONVERSION_RATE = "conversion_rate"  # Boolean success/failure
    CONTINUOUS = "continuous"  # Numeric values (savings, efficiency, etc.)
    COUNT = "count"  # Counting events
    DURATION = "duration"  # Time-based metrics


class ExperimentVariant(BaseModel):
    """
    A single variant in an A/B test.

    Represents one configuration/approach being tested.
    """

    name: str = Field(..., description="Variant name (e.g., 'control', 'treatment_a')")
    traffic_split: float = Field(..., ge=0.0, le=1.0, description="Percentage of traffic (0.0-1.0)")
    config: Dict[str, Any] = Field(..., description="Variant-specific configuration")
    description: Optional[str] = Field(None, description="What this variant tests")

    # Calculated fields
    is_control: bool = Field(False, description="Whether this is the control variant")

    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Ensure variant name is valid."""
        if not v or len(v) < 2:
            raise ValueError("Variant name must be at least 2 characters")
        # Only allow alphanumeric and underscores
        if not v.replace('_', '').isalnum():
            raise ValueError("Variant name must be alphanumeric with underscores only")
        return v.lower()

    @validator('traffic_split')
    def validate_traffic(cls, v: float) -> float:
        """Ensure traffic split is valid percentage."""
        if v < 0 or v > 1:
            raise ValueError("Traffic split must be between 0 and 1")
        return round(v, 4)  # Round to 4 decimal places


class Experiment(BaseModel):
    """
    An A/B/n experiment definition.

    Defines the experiment structure, variants, and success metrics.
    """

    experiment_id: Optional[str] = Field(None, description="Unique experiment identifier")
    name: str = Field(..., description="Human-readable experiment name")
    description: str = Field(..., description="What this experiment tests")
    hypothesis: str = Field(..., description="Expected outcome hypothesis")

    # Variants
    variants: List[ExperimentVariant] = Field(..., min_items=2, description="Experiment variants (A/B/C/n)")

    # Metrics
    primary_metric: str = Field(..., description="Primary success metric")
    primary_metric_type: MetricType = Field(..., description="Type of primary metric")
    secondary_metrics: List[str] = Field(default_factory=list, description="Additional tracking metrics")

    # Configuration
    min_sample_size: int = Field(100, ge=10, description="Minimum samples per variant")
    significance_level: float = Field(0.05, ge=0.01, le=0.1, description="Statistical significance threshold")
    power: float = Field(0.8, ge=0.5, le=0.99, description="Statistical power target")

    # Lifecycle
    status: ExperimentStatus = Field(ExperimentStatus.DRAFT, description="Current experiment status")
    start_date: Optional[datetime] = Field(None, description="When experiment started")
    end_date: Optional[datetime] = Field(None, description="When experiment ended")
    duration_days: Optional[int] = Field(None, ge=1, description="Planned duration in days")

    # Metadata
    created_by: str = Field(..., description="User who created experiment")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    tags: List[str] = Field(default_factory=list, description="Experiment tags for organization")

    @validator('variants')
    def validate_variants(cls, v: List[ExperimentVariant]) -> List[ExperimentVariant]:
        """Validate variants configuration."""
        if len(v) < 2:
            raise ValueError("Experiment must have at least 2 variants")

        # Check traffic splits sum to 1.0
        total_traffic = sum(variant.traffic_split for variant in v)
        if not (0.99 <= total_traffic <= 1.01):  # Allow small floating point error
            raise ValueError(f"Traffic splits must sum to 1.0, got {total_traffic}")

        # Ensure variant names are unique
        names = [variant.name for variant in v]
        if len(names) != len(set(names)):
            raise ValueError("Variant names must be unique")

        # Ensure exactly one control
        control_count = sum(1 for variant in v if variant.is_control)
        if control_count != 1:
            # Auto-assign first variant as control if none specified
            v[0].is_control = True

        return v

    @root_validator
    def generate_experiment_id(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unique experiment ID if not provided."""
        if not values.get('experiment_id'):
            # Create ID from name and timestamp
            name = values.get('name', 'experiment')
            timestamp = values.get('created_at', DeterministicClock.utcnow())
            id_string = f"{name}_{timestamp.isoformat()}"
            experiment_id = hashlib.sha256(id_string.encode()).hexdigest()[:16]
            values['experiment_id'] = experiment_id
        return values


class ExperimentMetrics(BaseModel):
    """
    Metrics collected for an experiment variant.

    Tracks performance metrics for statistical analysis.
    """

    variant_name: str = Field(..., description="Variant these metrics belong to")
    experiment_id: str = Field(..., description="Parent experiment ID")

    # Sample size
    sample_size: int = Field(0, ge=0, description="Number of observations")

    # Primary metric statistics
    primary_metric_name: str = Field(..., description="Name of primary metric")
    primary_metric_mean: float = Field(0.0, description="Mean value of primary metric")
    primary_metric_std: float = Field(0.0, ge=0, description="Standard deviation")
    primary_metric_median: Optional[float] = Field(None, description="Median value")

    # Conversion metrics (if applicable)
    conversions: Optional[int] = Field(None, ge=0, description="Number of successful conversions")
    conversion_rate: Optional[float] = Field(None, ge=0, le=1, description="Conversion rate")

    # Secondary metrics
    secondary_metrics: Dict[str, float] = Field(default_factory=dict, description="Secondary metric values")

    # Confidence intervals
    ci_lower: Optional[float] = Field(None, description="95% CI lower bound")
    ci_upper: Optional[float] = Field(None, description="95% CI upper bound")

    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")

    @validator('conversion_rate')
    def validate_conversion_rate(cls, v: Optional[float], values: Dict[str, Any]) -> Optional[float]:
        """Calculate conversion rate if not provided."""
        if v is None and values.get('conversions') is not None and values.get('sample_size', 0) > 0:
            v = values['conversions'] / values['sample_size']
        return v


class StatisticalSignificance(BaseModel):
    """
    Statistical significance test results.

    Results of comparing control vs. treatment variants.
    """

    experiment_id: str = Field(..., description="Experiment being analyzed")
    control_variant: str = Field(..., description="Control variant name")
    treatment_variant: str = Field(..., description="Treatment variant name")

    # Test results
    p_value: float = Field(..., ge=0, le=1, description="P-value from statistical test")
    is_significant: bool = Field(..., description="Whether result is statistically significant")
    significance_level: float = Field(0.05, description="Alpha threshold used")

    # Effect size
    effect_size: float = Field(..., description="Magnitude of effect (Cohen's d or similar)")
    relative_improvement: float = Field(..., description="% improvement over control")

    # Confidence intervals
    ci_lower: float = Field(..., description="95% CI lower bound for effect")
    ci_upper: float = Field(..., description="95% CI upper bound for effect")

    # Power analysis
    statistical_power: float = Field(..., ge=0, le=1, description="Statistical power of test")
    required_sample_size: int = Field(..., ge=0, description="Sample size needed for power target")

    # Recommendation
    recommendation: str = Field(..., description="Ship/iterate/continue testing")
    confidence: str = Field(..., description="high|medium|low confidence in result")

    analyzed_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")

    @validator('recommendation')
    def validate_recommendation(cls, v: str) -> str:
        """Ensure valid recommendation."""
        valid_recommendations = {'ship', 'iterate', 'continue', 'stop'}
        if v.lower() not in valid_recommendations:
            raise ValueError(f"Recommendation must be one of {valid_recommendations}")
        return v.lower()

    @validator('confidence')
    def validate_confidence(cls, v: str) -> str:
        """Ensure valid confidence level."""
        valid_levels = {'high', 'medium', 'low'}
        if v.lower() not in valid_levels:
            raise ValueError(f"Confidence must be one of {valid_levels}")
        return v.lower()


class ExperimentResult(BaseModel):
    """
    Complete experiment result including all variants and analysis.

    This is the final output of an experiment analysis.
    """

    experiment: Experiment = Field(..., description="Experiment configuration")
    metrics: List[ExperimentMetrics] = Field(..., description="Metrics for all variants")
    significance_tests: List[StatisticalSignificance] = Field(
        default_factory=list,
        description="Statistical tests for all variant pairs"
    )

    # Winner determination
    winner: Optional[str] = Field(None, description="Winning variant name (if any)")
    winner_confidence: Optional[str] = Field(None, description="Confidence in winner")
    winner_improvement: Optional[float] = Field(None, description="% improvement of winner")

    # Overall assessment
    total_samples: int = Field(0, ge=0, description="Total samples across all variants")
    is_conclusive: bool = Field(False, description="Whether experiment reached conclusion")
    days_running: Optional[int] = Field(None, ge=0, description="Days experiment has been running")

    # Final recommendation
    final_recommendation: str = Field(..., description="Overall recommendation for next steps")
    key_insights: List[str] = Field(default_factory=list, description="Key learnings from experiment")

    analyzed_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash for audit trail")

    @root_validator
    def calculate_provenance(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate SHA-256 hash for audit trail."""
        experiment_id = values.get('experiment', Experiment).experiment_id
        total_samples = values.get('total_samples', 0)
        analyzed_at = values.get('analyzed_at', DeterministicClock.utcnow())

        provenance_data = f"{experiment_id}|{total_samples}|{analyzed_at.isoformat()}"
        values['provenance_hash'] = hashlib.sha256(provenance_data.encode()).hexdigest()
        return values

    @root_validator
    def determine_winner(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Determine experiment winner from significance tests."""
        significance_tests = values.get('significance_tests', [])

        if not significance_tests:
            return values

        # Find the variant with the best performance and significance
        best_test = None
        best_improvement = float('-inf')

        for test in significance_tests:
            if test.is_significant and test.relative_improvement > best_improvement:
                best_improvement = test.relative_improvement
                best_test = test

        if best_test:
            values['winner'] = best_test.treatment_variant
            values['winner_confidence'] = best_test.confidence
            values['winner_improvement'] = best_improvement
            values['is_conclusive'] = True

        return values


class ExperimentAssignment(BaseModel):
    """
    User assignment to experiment variant.

    Tracks which variant a user was assigned to for consistency.
    """

    user_id: str = Field(..., description="User identifier")
    experiment_id: str = Field(..., description="Experiment identifier")
    variant_name: str = Field(..., description="Assigned variant name")
    assigned_at: datetime = Field(default_factory=datetime.utcnow, description="Assignment timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")

    # Ensure consistency
    assignment_hash: str = Field(..., description="Hash for deterministic assignment")

    @root_validator
    def calculate_assignment_hash(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate assignment hash for consistency."""
        user_id = values.get('user_id', '')
        experiment_id = values.get('experiment_id', '')
        hash_input = f"{user_id}:{experiment_id}"
        values['assignment_hash'] = hashlib.sha256(hash_input.encode()).hexdigest()
        return values
