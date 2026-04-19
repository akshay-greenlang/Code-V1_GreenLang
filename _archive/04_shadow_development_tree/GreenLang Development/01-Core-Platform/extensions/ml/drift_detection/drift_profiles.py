"""
Drift Profiles - Agent-Specific Drift Detection Configurations.

This module provides drift profile configurations for GreenLang Process Heat
agents GL-001 through GL-020. Each profile defines expected feature distributions,
drift thresholds, and alert configurations specific to the agent's domain.

Profiles:
    - GL001CarbonEmissionsDriftProfile: Carbon emissions calculation monitoring
    - GL003CSRDReportingDriftProfile: CSRD reporting compliance monitoring
    - GL006Scope3DriftProfile: Scope 3 emissions monitoring
    - GL010EmissionsGuardianDriftProfile: Real-time emissions guardian monitoring

Example:
    >>> from greenlang.ml.drift_detection import get_drift_profile
    >>> profile = get_drift_profile("GL-001")
    >>> print(f"Agent: {profile.agent_id}")
    >>> print(f"Features: {profile.expected_features}")
    >>> print(f"PSI Threshold: {profile.psi_threshold}")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field, validator


# =============================================================================
# Base Drift Profile
# =============================================================================

class FeatureSpec(BaseModel):
    """Specification for a single feature's expected distribution."""

    name: str = Field(..., description="Feature name")
    feature_type: str = Field(
        default="numerical",
        description="Feature type: numerical, categorical, binary"
    )

    # Expected distribution parameters (for numerical features)
    expected_mean: Optional[float] = Field(None, description="Expected mean value")
    expected_std: Optional[float] = Field(None, description="Expected standard deviation")
    expected_min: Optional[float] = Field(None, description="Expected minimum value")
    expected_max: Optional[float] = Field(None, description="Expected maximum value")

    # Drift sensitivity
    drift_sensitivity: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Drift sensitivity multiplier (higher = more sensitive)"
    )

    # Custom thresholds (override defaults)
    custom_psi_threshold: Optional[float] = Field(
        None, description="Custom PSI threshold for this feature"
    )
    custom_significance_level: Optional[float] = Field(
        None, description="Custom significance level for this feature"
    )

    # Feature importance for drift scoring
    importance_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Feature importance weight for drift scoring"
    )

    # Categorical feature specifics
    expected_categories: Optional[List[str]] = Field(
        None, description="Expected category values for categorical features"
    )
    category_proportions: Optional[Dict[str, float]] = Field(
        None, description="Expected category proportions"
    )


class AlertConfig(BaseModel):
    """Alert configuration for drift detection."""

    # Alert thresholds by severity
    low_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Low severity threshold"
    )
    medium_threshold: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Medium severity threshold"
    )
    high_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="High severity threshold"
    )
    critical_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Critical severity threshold"
    )

    # Alert behavior
    alert_on_low: bool = Field(default=False, description="Generate alerts for low severity")
    alert_on_medium: bool = Field(default=True, description="Generate alerts for medium severity")
    alert_on_high: bool = Field(default=True, description="Generate alerts for high severity")
    alert_on_critical: bool = Field(default=True, description="Generate alerts for critical severity")

    # Cooldown and throttling
    alert_cooldown_minutes: int = Field(
        default=60, ge=1, description="Minimum minutes between alerts"
    )
    max_alerts_per_day: int = Field(
        default=10, ge=1, le=100, description="Maximum alerts per day"
    )

    # Auto-remediation triggers
    trigger_retrain_on_high: bool = Field(
        default=False, description="Trigger retraining on high severity"
    )
    trigger_retrain_on_critical: bool = Field(
        default=True, description="Trigger retraining on critical severity"
    )
    trigger_rollback_on_critical: bool = Field(
        default=False, description="Trigger rollback on critical severity"
    )


class BaseDriftProfile(BaseModel):
    """
    Base drift profile for GreenLang Process Heat agents.

    This class defines the common configuration for drift detection
    across all agents, with agent-specific profiles inheriting and
    customizing these settings.

    Attributes:
        agent_id: Agent identifier (GL-001 through GL-020)
        agent_name: Human-readable agent name
        description: Profile description
        expected_features: List of expected feature specifications
        feature_names: List of feature names
        statistical_tests: List of statistical tests to run
        psi_threshold: Population Stability Index threshold
        significance_level: Statistical significance level
        alert_config: Alert configuration
    """

    # Agent identification
    agent_id: str = Field(..., description="Agent ID (GL-001 through GL-020)")
    agent_name: str = Field(..., description="Human-readable agent name")
    description: str = Field(..., description="Profile description")
    version: str = Field(default="1.0.0", description="Profile version")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Profile creation time"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Profile last update time"
    )

    # Feature specifications
    expected_features: List[FeatureSpec] = Field(
        default_factory=list, description="Expected feature specifications"
    )

    # Statistical test configuration
    statistical_tests: List[str] = Field(
        default=["ks_test", "psi", "js_divergence", "wasserstein"],
        description="Statistical tests to run"
    )
    primary_test: str = Field(
        default="ks_test", description="Primary statistical test"
    )

    # Drift thresholds
    psi_threshold: float = Field(
        default=0.2, ge=0.0, le=1.0, description="PSI threshold"
    )
    significance_level: float = Field(
        default=0.05, ge=0.001, le=0.1, description="Statistical significance level"
    )
    js_divergence_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Jensen-Shannon divergence threshold"
    )
    wasserstein_threshold: float = Field(
        default=0.1, ge=0.0, description="Wasserstein distance threshold"
    )

    # Drift share threshold
    drift_share_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Share of features needed to detect overall drift"
    )

    # Alert configuration
    alert_config: AlertConfig = Field(
        default_factory=AlertConfig, description="Alert configuration"
    )

    # Monitoring frequency
    monitoring_interval_minutes: int = Field(
        default=60, ge=1, description="Recommended monitoring interval"
    )
    min_samples_for_detection: int = Field(
        default=100, ge=30, description="Minimum samples for reliable drift detection"
    )

    # Domain-specific settings
    domain: str = Field(default="process_heat", description="Domain/application area")
    regulatory_framework: Optional[str] = Field(
        None, description="Applicable regulatory framework"
    )
    compliance_requirements: List[str] = Field(
        default_factory=list, description="Compliance requirements"
    )

    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [f.name for f in self.expected_features]

    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self.expected_features)

    def get_feature_spec(self, feature_name: str) -> Optional[FeatureSpec]:
        """Get feature specification by name."""
        for f in self.expected_features:
            if f.name == feature_name:
                return f
        return None

    def get_feature_threshold(self, feature_name: str) -> float:
        """Get PSI threshold for a feature (custom or default)."""
        spec = self.get_feature_spec(feature_name)
        if spec and spec.custom_psi_threshold is not None:
            return spec.custom_psi_threshold
        return self.psi_threshold

    def get_feature_significance(self, feature_name: str) -> float:
        """Get significance level for a feature (custom or default)."""
        spec = self.get_feature_spec(feature_name)
        if spec and spec.custom_significance_level is not None:
            return spec.custom_significance_level
        return self.significance_level

    def get_feature_weight(self, feature_name: str) -> float:
        """Get importance weight for a feature."""
        spec = self.get_feature_spec(feature_name)
        return spec.importance_weight if spec else 1.0

    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# Agent-Specific Drift Profiles
# =============================================================================

class GL001CarbonEmissionsDriftProfile(BaseDriftProfile):
    """
    Drift profile for GL-001 Carbon Emissions Calculator Agent.

    This agent calculates carbon emissions based on fuel consumption,
    energy usage, and activity data. Drift detection focuses on:
    - Fuel consumption patterns
    - Emission factors
    - Activity data distributions
    """

    def __init__(self, **kwargs):
        """Initialize GL-001 Carbon Emissions drift profile."""
        super().__init__(
            agent_id="GL-001",
            agent_name="Carbon Emissions Calculator",
            description=(
                "Drift profile for carbon emissions calculation agent. "
                "Monitors fuel consumption, emission factors, and activity data "
                "for regulatory compliance."
            ),
            expected_features=[
                FeatureSpec(
                    name="fuel_consumption_mwh",
                    feature_type="numerical",
                    expected_mean=500.0,
                    expected_std=200.0,
                    expected_min=0.0,
                    expected_max=5000.0,
                    importance_weight=2.0,
                    drift_sensitivity=1.5,
                ),
                FeatureSpec(
                    name="emission_factor",
                    feature_type="numerical",
                    expected_mean=0.5,
                    expected_std=0.1,
                    expected_min=0.0,
                    expected_max=2.0,
                    importance_weight=2.5,
                    custom_psi_threshold=0.15,  # More sensitive for emission factors
                ),
                FeatureSpec(
                    name="activity_data",
                    feature_type="numerical",
                    expected_mean=1000.0,
                    expected_std=500.0,
                    expected_min=0.0,
                    importance_weight=1.5,
                ),
                FeatureSpec(
                    name="temperature_celsius",
                    feature_type="numerical",
                    expected_mean=25.0,
                    expected_std=15.0,
                    expected_min=-20.0,
                    expected_max=60.0,
                    importance_weight=1.0,
                ),
                FeatureSpec(
                    name="fuel_type",
                    feature_type="categorical",
                    expected_categories=["natural_gas", "coal", "oil", "biomass", "electricity"],
                    importance_weight=1.5,
                ),
                FeatureSpec(
                    name="scope",
                    feature_type="categorical",
                    expected_categories=["scope1", "scope2", "scope3"],
                    importance_weight=1.0,
                ),
            ],
            psi_threshold=0.2,
            significance_level=0.05,
            drift_share_threshold=0.25,
            monitoring_interval_minutes=30,
            min_samples_for_detection=200,
            regulatory_framework="GHG Protocol",
            compliance_requirements=[
                "ISO 14064-1",
                "GHG Protocol Corporate Standard",
                "IPCC Guidelines",
            ],
            alert_config=AlertConfig(
                low_threshold=0.1,
                medium_threshold=0.2,
                high_threshold=0.3,
                critical_threshold=0.5,
                alert_on_medium=True,
                trigger_retrain_on_critical=True,
            ),
            **kwargs
        )


class GL003CSRDReportingDriftProfile(BaseDriftProfile):
    """
    Drift profile for GL-003 CSRD Reporting Agent.

    This agent handles Corporate Sustainability Reporting Directive
    compliance. Drift detection focuses on:
    - Environmental metrics
    - Social indicators
    - Governance data
    - Materiality assessments
    """

    def __init__(self, **kwargs):
        """Initialize GL-003 CSRD Reporting drift profile."""
        super().__init__(
            agent_id="GL-003",
            agent_name="CSRD Reporting Agent",
            description=(
                "Drift profile for CSRD reporting compliance agent. "
                "Monitors ESG metrics, materiality scores, and disclosure data "
                "for EU regulatory compliance."
            ),
            expected_features=[
                FeatureSpec(
                    name="environmental_score",
                    feature_type="numerical",
                    expected_mean=65.0,
                    expected_std=20.0,
                    expected_min=0.0,
                    expected_max=100.0,
                    importance_weight=2.0,
                ),
                FeatureSpec(
                    name="social_score",
                    feature_type="numerical",
                    expected_mean=60.0,
                    expected_std=15.0,
                    expected_min=0.0,
                    expected_max=100.0,
                    importance_weight=1.8,
                ),
                FeatureSpec(
                    name="governance_score",
                    feature_type="numerical",
                    expected_mean=70.0,
                    expected_std=15.0,
                    expected_min=0.0,
                    expected_max=100.0,
                    importance_weight=1.8,
                ),
                FeatureSpec(
                    name="materiality_assessment",
                    feature_type="numerical",
                    expected_mean=0.7,
                    expected_std=0.2,
                    expected_min=0.0,
                    expected_max=1.0,
                    importance_weight=2.5,
                    custom_psi_threshold=0.1,  # Highly sensitive
                ),
                FeatureSpec(
                    name="disclosure_completeness",
                    feature_type="numerical",
                    expected_mean=0.85,
                    expected_std=0.1,
                    expected_min=0.0,
                    expected_max=1.0,
                    importance_weight=2.0,
                ),
                FeatureSpec(
                    name="data_quality_score",
                    feature_type="numerical",
                    expected_mean=0.9,
                    expected_std=0.1,
                    expected_min=0.0,
                    expected_max=1.0,
                    importance_weight=1.5,
                ),
                FeatureSpec(
                    name="reporting_period",
                    feature_type="categorical",
                    expected_categories=["Q1", "Q2", "Q3", "Q4", "annual"],
                    importance_weight=0.5,
                ),
                FeatureSpec(
                    name="sector",
                    feature_type="categorical",
                    expected_categories=[
                        "energy", "manufacturing", "transportation",
                        "construction", "services", "other"
                    ],
                    importance_weight=1.0,
                ),
            ],
            psi_threshold=0.15,  # More stringent for regulatory reporting
            significance_level=0.05,
            drift_share_threshold=0.2,
            monitoring_interval_minutes=60,
            min_samples_for_detection=100,
            regulatory_framework="EU CSRD",
            compliance_requirements=[
                "EU CSRD",
                "ESRS Standards",
                "Double Materiality Assessment",
                "EFRAG Guidelines",
            ],
            alert_config=AlertConfig(
                low_threshold=0.08,
                medium_threshold=0.15,
                high_threshold=0.25,
                critical_threshold=0.4,
                alert_on_low=True,  # Higher sensitivity for regulatory
                alert_on_medium=True,
                trigger_retrain_on_high=True,  # Earlier retraining trigger
                trigger_retrain_on_critical=True,
            ),
            **kwargs
        )


class GL006Scope3DriftProfile(BaseDriftProfile):
    """
    Drift profile for GL-006 Scope 3 Emissions Agent.

    This agent handles Scope 3 (value chain) emissions calculations
    across all 15 categories. Drift detection focuses on:
    - Category-specific emission factors
    - Spend-based calculations
    - Activity-based calculations
    - Supplier data quality
    """

    def __init__(self, **kwargs):
        """Initialize GL-006 Scope 3 drift profile."""
        super().__init__(
            agent_id="GL-006",
            agent_name="Scope 3 Emissions Agent",
            description=(
                "Drift profile for Scope 3 emissions calculation agent. "
                "Monitors value chain emissions across all 15 GHG Protocol categories "
                "with supplier and activity data tracking."
            ),
            expected_features=[
                FeatureSpec(
                    name="category_1_purchased_goods",
                    feature_type="numerical",
                    expected_mean=50000.0,
                    expected_std=25000.0,
                    expected_min=0.0,
                    importance_weight=2.0,
                ),
                FeatureSpec(
                    name="category_2_capital_goods",
                    feature_type="numerical",
                    expected_mean=20000.0,
                    expected_std=15000.0,
                    expected_min=0.0,
                    importance_weight=1.5,
                ),
                FeatureSpec(
                    name="category_3_fuel_energy",
                    feature_type="numerical",
                    expected_mean=15000.0,
                    expected_std=10000.0,
                    expected_min=0.0,
                    importance_weight=1.8,
                ),
                FeatureSpec(
                    name="category_4_upstream_transport",
                    feature_type="numerical",
                    expected_mean=8000.0,
                    expected_std=5000.0,
                    expected_min=0.0,
                    importance_weight=1.5,
                ),
                FeatureSpec(
                    name="category_5_waste",
                    feature_type="numerical",
                    expected_mean=3000.0,
                    expected_std=2000.0,
                    expected_min=0.0,
                    importance_weight=1.2,
                ),
                FeatureSpec(
                    name="category_6_business_travel",
                    feature_type="numerical",
                    expected_mean=5000.0,
                    expected_std=3000.0,
                    expected_min=0.0,
                    importance_weight=1.0,
                ),
                FeatureSpec(
                    name="category_7_commuting",
                    feature_type="numerical",
                    expected_mean=4000.0,
                    expected_std=2000.0,
                    expected_min=0.0,
                    importance_weight=1.0,
                ),
                FeatureSpec(
                    name="supplier_data_quality",
                    feature_type="numerical",
                    expected_mean=0.6,
                    expected_std=0.2,
                    expected_min=0.0,
                    expected_max=1.0,
                    importance_weight=2.5,
                    custom_psi_threshold=0.15,
                ),
                FeatureSpec(
                    name="spend_amount_usd",
                    feature_type="numerical",
                    expected_mean=1000000.0,
                    expected_std=500000.0,
                    expected_min=0.0,
                    importance_weight=1.8,
                ),
                FeatureSpec(
                    name="emission_factor_source",
                    feature_type="categorical",
                    expected_categories=[
                        "supplier_specific", "industry_average",
                        "spend_based", "hybrid"
                    ],
                    importance_weight=1.5,
                ),
                FeatureSpec(
                    name="calculation_method",
                    feature_type="categorical",
                    expected_categories=[
                        "spend_based", "activity_based",
                        "average_data", "supplier_specific"
                    ],
                    importance_weight=1.5,
                ),
            ],
            psi_threshold=0.2,
            significance_level=0.05,
            drift_share_threshold=0.25,
            monitoring_interval_minutes=60,
            min_samples_for_detection=150,
            regulatory_framework="GHG Protocol Scope 3",
            compliance_requirements=[
                "GHG Protocol Scope 3 Standard",
                "SBTi Scope 3 Target Setting",
                "CDP Supply Chain Program",
            ],
            alert_config=AlertConfig(
                low_threshold=0.1,
                medium_threshold=0.2,
                high_threshold=0.35,
                critical_threshold=0.5,
                alert_on_medium=True,
                trigger_retrain_on_critical=True,
            ),
            **kwargs
        )


class GL010EmissionsGuardianDriftProfile(BaseDriftProfile):
    """
    Drift profile for GL-010 Emissions Guardian Agent.

    This agent provides real-time emissions monitoring and anomaly
    detection. Drift detection focuses on:
    - Real-time sensor data
    - Threshold violations
    - Pattern deviations
    - Predictive maintenance signals
    """

    def __init__(self, **kwargs):
        """Initialize GL-010 Emissions Guardian drift profile."""
        super().__init__(
            agent_id="GL-010",
            agent_name="Emissions Guardian Agent",
            description=(
                "Drift profile for real-time emissions monitoring guardian. "
                "Monitors sensor data, threshold violations, and anomaly patterns "
                "for immediate alerting and intervention."
            ),
            expected_features=[
                FeatureSpec(
                    name="real_time_emissions_rate",
                    feature_type="numerical",
                    expected_mean=100.0,
                    expected_std=30.0,
                    expected_min=0.0,
                    expected_max=500.0,
                    importance_weight=3.0,  # Highest priority
                    custom_psi_threshold=0.1,  # Very sensitive
                    drift_sensitivity=2.0,
                ),
                FeatureSpec(
                    name="sensor_reading_temperature",
                    feature_type="numerical",
                    expected_mean=150.0,
                    expected_std=50.0,
                    expected_min=0.0,
                    expected_max=500.0,
                    importance_weight=2.0,
                ),
                FeatureSpec(
                    name="sensor_reading_pressure",
                    feature_type="numerical",
                    expected_mean=2.0,
                    expected_std=0.5,
                    expected_min=0.0,
                    expected_max=10.0,
                    importance_weight=2.0,
                ),
                FeatureSpec(
                    name="sensor_reading_flow_rate",
                    feature_type="numerical",
                    expected_mean=50.0,
                    expected_std=20.0,
                    expected_min=0.0,
                    expected_max=200.0,
                    importance_weight=2.0,
                ),
                FeatureSpec(
                    name="anomaly_score",
                    feature_type="numerical",
                    expected_mean=0.1,
                    expected_std=0.15,
                    expected_min=0.0,
                    expected_max=1.0,
                    importance_weight=2.5,
                    custom_psi_threshold=0.12,
                ),
                FeatureSpec(
                    name="threshold_proximity",
                    feature_type="numerical",
                    expected_mean=0.6,
                    expected_std=0.2,
                    expected_min=0.0,
                    expected_max=1.0,
                    importance_weight=2.0,
                ),
                FeatureSpec(
                    name="equipment_efficiency",
                    feature_type="numerical",
                    expected_mean=0.85,
                    expected_std=0.1,
                    expected_min=0.0,
                    expected_max=1.0,
                    importance_weight=1.5,
                ),
                FeatureSpec(
                    name="maintenance_score",
                    feature_type="numerical",
                    expected_mean=0.9,
                    expected_std=0.1,
                    expected_min=0.0,
                    expected_max=1.0,
                    importance_weight=1.5,
                ),
                FeatureSpec(
                    name="alert_status",
                    feature_type="categorical",
                    expected_categories=["normal", "warning", "critical", "emergency"],
                    importance_weight=2.5,
                ),
                FeatureSpec(
                    name="equipment_type",
                    feature_type="categorical",
                    expected_categories=[
                        "boiler", "furnace", "heat_exchanger",
                        "compressor", "pump", "other"
                    ],
                    importance_weight=1.0,
                ),
            ],
            psi_threshold=0.1,  # Very sensitive for real-time monitoring
            significance_level=0.01,  # Stricter significance level
            drift_share_threshold=0.15,  # Lower threshold for critical system
            monitoring_interval_minutes=5,  # Frequent monitoring
            min_samples_for_detection=50,  # Faster detection with fewer samples
            regulatory_framework="EPA Continuous Monitoring",
            compliance_requirements=[
                "EPA CEMS Requirements",
                "Industrial Emissions Directive",
                "Process Safety Management",
            ],
            alert_config=AlertConfig(
                low_threshold=0.05,
                medium_threshold=0.1,
                high_threshold=0.2,
                critical_threshold=0.3,
                alert_on_low=True,  # All severity levels trigger alerts
                alert_on_medium=True,
                alert_on_high=True,
                alert_on_critical=True,
                alert_cooldown_minutes=15,  # Short cooldown
                trigger_retrain_on_high=True,
                trigger_retrain_on_critical=True,
                trigger_rollback_on_critical=True,  # Auto-rollback enabled
            ),
            **kwargs
        )


# =============================================================================
# Additional Agent Profiles (GL-002 through GL-020 templates)
# =============================================================================

class GL002CBAMComplianceProfile(BaseDriftProfile):
    """Drift profile for GL-002 CBAM Compliance Agent."""

    def __init__(self, **kwargs):
        super().__init__(
            agent_id="GL-002",
            agent_name="CBAM Compliance Agent",
            description="Carbon Border Adjustment Mechanism compliance monitoring",
            expected_features=[
                FeatureSpec(name="embedded_emissions", feature_type="numerical", importance_weight=2.5),
                FeatureSpec(name="import_quantity", feature_type="numerical", importance_weight=2.0),
                FeatureSpec(name="carbon_price", feature_type="numerical", importance_weight=2.0),
                FeatureSpec(name="product_code", feature_type="categorical", importance_weight=1.5),
                FeatureSpec(name="country_of_origin", feature_type="categorical", importance_weight=1.5),
            ],
            psi_threshold=0.15,
            regulatory_framework="EU CBAM",
            **kwargs
        )


class GL004EUDRComplianceProfile(BaseDriftProfile):
    """Drift profile for GL-004 EUDR Compliance Agent."""

    def __init__(self, **kwargs):
        super().__init__(
            agent_id="GL-004",
            agent_name="EUDR Compliance Agent",
            description="EU Deforestation Regulation compliance monitoring",
            expected_features=[
                FeatureSpec(name="deforestation_risk_score", feature_type="numerical", importance_weight=3.0),
                FeatureSpec(name="geolocation_accuracy", feature_type="numerical", importance_weight=2.5),
                FeatureSpec(name="supply_chain_visibility", feature_type="numerical", importance_weight=2.0),
                FeatureSpec(name="commodity_type", feature_type="categorical", importance_weight=2.0),
            ],
            psi_threshold=0.1,  # Very sensitive for deforestation monitoring
            regulatory_framework="EU EUDR",
            **kwargs
        )


class GL005BuildingEnergyProfile(BaseDriftProfile):
    """Drift profile for GL-005 Building Energy Agent."""

    def __init__(self, **kwargs):
        super().__init__(
            agent_id="GL-005",
            agent_name="Building Energy Agent",
            description="Building energy consumption and efficiency monitoring",
            expected_features=[
                FeatureSpec(name="energy_consumption_kwh", feature_type="numerical", importance_weight=2.0),
                FeatureSpec(name="building_efficiency_score", feature_type="numerical", importance_weight=2.0),
                FeatureSpec(name="hvac_load", feature_type="numerical", importance_weight=1.8),
                FeatureSpec(name="lighting_load", feature_type="numerical", importance_weight=1.5),
                FeatureSpec(name="building_type", feature_type="categorical", importance_weight=1.0),
            ],
            psi_threshold=0.2,
            **kwargs
        )


class GL007TaxonomyAlignmentProfile(BaseDriftProfile):
    """Drift profile for GL-007 EU Taxonomy Agent."""

    def __init__(self, **kwargs):
        super().__init__(
            agent_id="GL-007",
            agent_name="EU Taxonomy Agent",
            description="EU Taxonomy alignment assessment monitoring",
            expected_features=[
                FeatureSpec(name="taxonomy_alignment_score", feature_type="numerical", importance_weight=2.5),
                FeatureSpec(name="dnsh_compliance", feature_type="numerical", importance_weight=2.0),
                FeatureSpec(name="minimum_safeguards", feature_type="numerical", importance_weight=2.0),
                FeatureSpec(name="economic_activity", feature_type="categorical", importance_weight=1.5),
            ],
            psi_threshold=0.15,
            regulatory_framework="EU Taxonomy",
            **kwargs
        )


# =============================================================================
# Profile Registry and Factory
# =============================================================================

# Registry of all drift profiles
DRIFT_PROFILE_REGISTRY: Dict[str, Type[BaseDriftProfile]] = {
    "GL-001": GL001CarbonEmissionsDriftProfile,
    "GL-002": GL002CBAMComplianceProfile,
    "GL-003": GL003CSRDReportingDriftProfile,
    "GL-004": GL004EUDRComplianceProfile,
    "GL-005": GL005BuildingEnergyProfile,
    "GL-006": GL006Scope3DriftProfile,
    "GL-007": GL007TaxonomyAlignmentProfile,
    "GL-010": GL010EmissionsGuardianDriftProfile,
}


def get_drift_profile(agent_id: str, **kwargs) -> BaseDriftProfile:
    """
    Get drift profile for an agent.

    Args:
        agent_id: Agent identifier (GL-001 through GL-020).
        **kwargs: Additional arguments to pass to profile constructor.

    Returns:
        Drift profile for the specified agent.

    Raises:
        ValueError: If agent_id is not supported.

    Example:
        >>> profile = get_drift_profile("GL-001")
        >>> print(f"Agent: {profile.agent_name}")
        >>> print(f"Features: {profile.feature_names}")
    """
    if agent_id in DRIFT_PROFILE_REGISTRY:
        profile_class = DRIFT_PROFILE_REGISTRY[agent_id]
        return profile_class(**kwargs)

    # For agents without specific profiles, create a default profile
    return BaseDriftProfile(
        agent_id=agent_id,
        agent_name=f"Process Heat Agent {agent_id}",
        description=f"Default drift profile for {agent_id}",
        **kwargs
    )


def list_available_profiles() -> List[str]:
    """
    List all available drift profiles.

    Returns:
        List of agent IDs with specific drift profiles.
    """
    return list(DRIFT_PROFILE_REGISTRY.keys())


def create_custom_profile(
    agent_id: str,
    agent_name: str,
    features: List[Dict[str, Any]],
    **kwargs
) -> BaseDriftProfile:
    """
    Create a custom drift profile.

    Args:
        agent_id: Agent identifier.
        agent_name: Human-readable agent name.
        features: List of feature specification dictionaries.
        **kwargs: Additional profile configuration.

    Returns:
        Custom BaseDriftProfile.

    Example:
        >>> profile = create_custom_profile(
        ...     agent_id="GL-015",
        ...     agent_name="Custom Agent",
        ...     features=[
        ...         {"name": "feature1", "feature_type": "numerical", "importance_weight": 2.0},
        ...         {"name": "feature2", "feature_type": "categorical"},
        ...     ]
        ... )
    """
    feature_specs = [FeatureSpec(**f) for f in features]

    return BaseDriftProfile(
        agent_id=agent_id,
        agent_name=agent_name,
        description=kwargs.pop("description", f"Custom profile for {agent_name}"),
        expected_features=feature_specs,
        **kwargs
    )
