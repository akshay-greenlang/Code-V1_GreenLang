"""
Configuration module for AGENT-MRV-015 Capital Goods Category 2 Agent.

This module provides thread-safe singleton configuration management for calculating
Scope 3 Category 2 emissions from capital goods purchases. Supports spend-based,
supplier-specific, average-data, and hybrid calculation methods with EEIO factors,
capitalization policies, useful life allocation, and compliance with GHG Protocol
Scope 3 Standard.

Environment Variables:
    All configuration uses GL_CG_* prefix for capital goods service.

Example:
    >>> config = CapitalGoodsConfig.from_env()
    >>> config.validate()
    >>> print(config.calculation.default_method)
    'hybrid'

Thread Safety:
    This module uses threading.RLock for singleton pattern thread safety.

Author: GreenLang Backend Team
Version: 1.0.0
"""

import os
import threading
import logging
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum

# Logger setup
logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class CalculationMethod(str, Enum):
    """Capital goods calculation methods."""

    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"


class GWPSource(str, Enum):
    """GWP value sources."""

    AR5 = "AR5"
    AR6 = "AR6"
    AR4 = "AR4"


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification methods."""

    MONTE_CARLO = "monte_carlo"
    ANALYTICAL = "analytical"
    BOOTSTRAP = "bootstrap"


class CapitalizationPolicy(str, Enum):
    """Asset capitalization policies."""

    COMPANY_DEFINED = "COMPANY_DEFINED"
    GAAP = "GAAP"
    IFRS = "IFRS"
    TAX_BASIS = "TAX_BASIS"
    MATERIALITY_BASED = "MATERIALITY_BASED"


class RegulatoryFramework(str, Enum):
    """Supported regulatory frameworks."""

    GHG_PROTOCOL_SCOPE3 = "GHG_PROTOCOL_SCOPE3"
    ISO_14064_1 = "ISO_14064_1"
    CSRD_ESRS_E1 = "CSRD_ESRS_E1"
    CDP_SUPPLY_CHAIN = "CDP_SUPPLY_CHAIN"
    SBTI_SCOPE3 = "SBTI_SCOPE3"
    PCAF_INVESTMENTS = "PCAF_INVESTMENTS"
    GRI_305 = "GRI_305"


class AssetCategory(str, Enum):
    """Capital goods asset categories."""

    BUILDINGS = "buildings"
    MACHINERY = "machinery"
    VEHICLES = "vehicles"
    IT_EQUIPMENT = "it_equipment"
    FURNITURE = "furniture"
    INFRASTRUCTURE = "infrastructure"
    TOOLS = "tools"
    OTHER = "other"


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    name: str = "greenlang"
    user: str = "greenlang"
    password: str = ""
    pool_min: int = 2
    pool_max: int = 10
    ssl_mode: str = "prefer"
    schema: str = "capital_goods_service"
    connection_timeout: int = 30
    command_timeout: int = 60

    def get_dsn(self) -> str:
        """Get PostgreSQL DSN."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.name}"
            f"?sslmode={self.ssl_mode}"
        )


@dataclass
class CalculationConfig:
    """Capital goods calculation configuration."""

    # Method selection
    default_method: str = "hybrid"
    default_gwp_source: str = "AR6"

    # Precision
    decimal_precision: int = 10
    rounding_mode: str = "ROUND_HALF_UP"

    # Method enablement
    enable_spend_based: bool = True
    enable_average_data: bool = True
    enable_supplier_specific: bool = True
    enable_hybrid: bool = True

    # EEIO configuration
    eeio_fallback_digits: int = 2
    cpi_base_year: int = 2021
    default_currency: str = "USD"

    # Boundary adjustments
    enable_margin_removal: bool = True
    enable_transport_removal: bool = True
    default_margin_rate: Decimal = Decimal("0.30")
    default_transport_rate: Decimal = Decimal("0.05")

    # Capitalization
    capitalization_threshold_usd: Decimal = Decimal("5000")
    default_capitalization_policy: str = "COMPANY_DEFINED"
    min_useful_life_years: int = 1
    max_useful_life_years: int = 100

    # Asset classification
    enable_asset_classification: bool = True
    enable_subcategory_resolution: bool = True

    # Allocation
    enable_useful_life_allocation: bool = True
    enable_first_year_convention: bool = True
    first_year_fraction: Decimal = Decimal("1.0")

    # CapEx volatility
    enable_capex_volatility_context: bool = True
    volatility_ratio_threshold: Decimal = Decimal("2.0")
    rolling_average_years: int = 3

    # Leased assets
    enable_leased_asset_exclusion: bool = True

    # Data quality
    require_purchase_date: bool = True
    require_useful_life: bool = True
    allow_category_fallback: bool = True

    # Double counting prevention
    enable_double_counting_check: bool = True
    enable_depreciation_exclusion: bool = True


@dataclass
class UncertaintyConfig:
    """Uncertainty quantification configuration."""

    enable: bool = True
    method: str = "monte_carlo"
    iterations: int = 5000
    confidence_levels: List[int] = field(default_factory=lambda: [90, 95, 99])
    seed: Optional[int] = 42

    # Coefficient of variation by method
    cv_spend_based: Decimal = Decimal("0.50")
    cv_average_data: Decimal = Decimal("0.35")
    cv_supplier_specific: Decimal = Decimal("0.15")
    cv_hybrid: Decimal = Decimal("0.25")

    # EEIO uncertainty
    cv_eeio_sector_match: Decimal = Decimal("0.40")
    cv_eeio_fallback: Decimal = Decimal("0.65")

    # Temporal uncertainty
    cv_inflation_adjustment: Decimal = Decimal("0.08")
    cv_useful_life: Decimal = Decimal("0.20")

    # Enable components
    enable_parameter_uncertainty: bool = True
    enable_model_uncertainty: bool = True
    enable_scenario_uncertainty: bool = False

    # Distribution types
    activity_distribution: str = "lognormal"
    ef_distribution: str = "lognormal"

    # Sensitivity analysis
    enable_sensitivity_analysis: bool = True
    sensitivity_parameters: List[str] = field(default_factory=lambda: [
        "spend_amount",
        "emission_factor",
        "useful_life",
        "margin_rate",
        "transport_rate"
    ])


@dataclass
class ComplianceConfig:
    """Regulatory compliance configuration."""

    enabled_frameworks: List[str] = field(default_factory=lambda: [
        "GHG_PROTOCOL_SCOPE3",
        "ISO_14064_1",
        "CSRD_ESRS_E1",
        "CDP_SUPPLY_CHAIN",
        "SBTI_SCOPE3",
        "PCAF_INVESTMENTS",
        "GRI_305"
    ])

    # Version tracking
    ghg_protocol_version: str = "2011"
    iso_14064_version: str = "2018"
    csrd_version: str = "2023"
    cdp_version: str = "2024"
    sbti_version: str = "2.0"
    pcaf_version: str = "2022"
    gri_version: str = "2021"

    # Compliance rules
    enable_double_counting_check: bool = True
    require_base_year: bool = True
    require_boundary_definition: bool = True
    require_consolidation_approach: bool = True

    # GHG Protocol specific
    ghg_protocol_require_category_2: bool = True
    ghg_protocol_require_useful_life: bool = True
    ghg_protocol_allow_no_allocation: bool = False

    # CSRD specific
    csrd_require_value_chain_mapping: bool = True
    csrd_require_double_materiality: bool = True

    # SBTi specific
    sbti_require_upstream_coverage: bool = True
    sbti_minimum_coverage_percent: Decimal = Decimal("67.0")

    # PCAF specific
    pcaf_enable_data_quality_score: bool = True
    pcaf_score_range: tuple = (1, 5)

    # Reporting thresholds
    materiality_threshold_tco2e: Optional[Decimal] = None
    reporting_currency: str = "USD"

    # Validation
    enable_compliance_validation: bool = True
    fail_on_non_compliance: bool = False


@dataclass
class DQIConfig:
    """Data Quality Indicator configuration."""

    enable: bool = True

    # DQI weights (must sum to 1.0)
    weight_reliability: Decimal = Decimal("0.25")
    weight_completeness: Decimal = Decimal("0.25")
    weight_temporal: Decimal = Decimal("0.20")
    weight_geographical: Decimal = Decimal("0.15")
    weight_technological: Decimal = Decimal("0.15")

    # Quality thresholds
    quality_threshold_excellent: Decimal = Decimal("4.0")
    quality_threshold_good: Decimal = Decimal("3.0")
    quality_threshold_fair: Decimal = Decimal("2.0")
    quality_threshold_poor: Decimal = Decimal("1.0")

    minimum_acceptable_score: Decimal = Decimal("2.5")

    # Scoring criteria
    supplier_specific_score: Decimal = Decimal("5.0")
    average_data_score: Decimal = Decimal("3.0")
    spend_based_score: Decimal = Decimal("2.0")

    # Temporal scoring
    temporal_max_age_years: int = 5
    temporal_score_current_year: Decimal = Decimal("5.0")
    temporal_score_one_year: Decimal = Decimal("4.5")
    temporal_score_two_year: Decimal = Decimal("4.0")
    temporal_score_three_year: Decimal = Decimal("3.0")
    temporal_score_old: Decimal = Decimal("2.0")

    # Geographical scoring
    geo_score_specific: Decimal = Decimal("5.0")
    geo_score_regional: Decimal = Decimal("4.0")
    geo_score_continental: Decimal = Decimal("3.0")
    geo_score_global: Decimal = Decimal("2.0")

    # Technological scoring
    tech_score_specific: Decimal = Decimal("5.0")
    tech_score_sector_average: Decimal = Decimal("3.0")
    tech_score_generic: Decimal = Decimal("2.0")

    # PCAF mapping
    enable_pcaf_mapping: bool = True
    dqi_to_pcaf_mapping: Dict[str, int] = field(default_factory=lambda: {
        "excellent": 1,
        "good": 2,
        "fair": 3,
        "poor": 4,
        "very_poor": 5
    })


@dataclass
class APIConfig:
    """API configuration."""

    prefix: str = "/api/v1/capital-goods"
    max_batch_size: int = 500
    pagination_default: int = 100
    pagination_max: int = 1000

    # Rate limiting
    rate_limit_requests: int = 60
    rate_limit_period: int = 60

    # Caching
    cache_ttl: int = 300
    enable_cache: bool = True

    # Timeouts
    request_timeout: int = 30
    calculation_timeout: int = 120
    batch_timeout: int = 300

    # Response options
    include_provenance: bool = True
    include_uncertainty: bool = True
    include_dqi: bool = True
    include_compliance: bool = True

    # Error handling
    return_partial_results: bool = False
    max_error_detail_length: int = 500


@dataclass
class MetricsConfig:
    """Observability metrics configuration."""

    enable: bool = True
    prefix: str = "gl_cg"
    namespace: str = "greenlang"
    subsystem: str = "capital_goods"

    # Histogram buckets (processing time in seconds)
    histogram_buckets: List[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])

    # Counter metrics
    enable_calculation_counter: bool = True
    enable_error_counter: bool = True
    enable_validation_counter: bool = True

    # Gauge metrics
    enable_active_calculations: bool = True
    enable_cache_size: bool = True

    # Histogram metrics
    enable_processing_time: bool = True
    enable_emission_distribution: bool = True
    enable_dqi_distribution: bool = True

    # Labels
    default_labels: List[str] = field(default_factory=lambda: [
        "method",
        "asset_category",
        "framework",
        "status"
    ])


@dataclass
class ProvenanceConfig:
    """Data provenance and audit trail configuration."""

    enable: bool = True
    hash_algorithm: str = "sha256"
    chain_hashing: bool = True

    # Provenance stages
    stage_names: List[str] = field(default_factory=lambda: [
        "input_validation",
        "capitalization_check",
        "asset_classification",
        "useful_life_determination",
        "method_selection",
        "spend_normalization",
        "margin_removal",
        "transport_removal",
        "eeio_matching",
        "emission_factor_lookup",
        "emission_calculation",
        "useful_life_allocation",
        "uncertainty_quantification",
        "dqi_calculation",
        "compliance_validation",
        "aggregation",
        "output_generation"
    ])

    # Storage
    store_intermediate_results: bool = True
    compress_provenance: bool = True

    # Retention
    retention_days: int = 2555
    auto_cleanup: bool = True

    # Audit
    track_user_actions: bool = True
    track_data_lineage: bool = True
    track_assumption_changes: bool = True


@dataclass
class AssetConfig:
    """Asset classification and management configuration."""

    enable_asset_classification: bool = True
    enable_subcategory_resolution: bool = True

    # Default useful life by category (years)
    default_useful_life: int = 10
    useful_life_buildings: int = 40
    useful_life_machinery: int = 15
    useful_life_vehicles: int = 8
    useful_life_it_equipment: int = 5
    useful_life_furniture: int = 10
    useful_life_infrastructure: int = 50
    useful_life_tools: int = 7
    useful_life_other: int = 10

    # Capitalization thresholds by category (USD)
    threshold_buildings: Decimal = Decimal("50000")
    threshold_machinery: Decimal = Decimal("10000")
    threshold_vehicles: Decimal = Decimal("5000")
    threshold_it_equipment: Decimal = Decimal("1000")
    threshold_furniture: Decimal = Decimal("2500")
    threshold_infrastructure: Decimal = Decimal("100000")
    threshold_tools: Decimal = Decimal("500")
    threshold_other: Decimal = Decimal("5000")

    # Leased asset handling
    enable_leased_asset_exclusion: bool = True
    leased_asset_categories: List[str] = field(default_factory=lambda: [
        "finance_lease",
        "operating_lease",
        "rental"
    ])

    # Depreciation methods
    depreciation_method_straight_line: bool = True
    depreciation_method_declining_balance: bool = True
    depreciation_method_units_of_production: bool = True

    # Asset tracking
    require_asset_id: bool = True
    require_purchase_order: bool = False
    require_supplier_info: bool = False

    # Maximum constraints
    max_useful_life: int = 100
    min_useful_life: int = 1


# ============================================================================
# Main Configuration Class
# ============================================================================


class CapitalGoodsConfig:
    """
    Thread-safe singleton configuration for Capital Goods Category 2 Agent.

    This class manages all configuration settings for calculating Scope 3
    Category 2 emissions from capital goods purchases. It implements the
    singleton pattern with thread safety using RLock.

    Example:
        >>> config = CapitalGoodsConfig.from_env()
        >>> config.validate()
        >>> print(config.calculation.default_method)
        'hybrid'

    Thread Safety:
        Uses threading.RLock for thread-safe singleton implementation.
    """

    _instance: Optional['CapitalGoodsConfig'] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(
        self,
        database: DatabaseConfig,
        calculation: CalculationConfig,
        uncertainty: UncertaintyConfig,
        compliance: ComplianceConfig,
        dqi: DQIConfig,
        api: APIConfig,
        metrics: MetricsConfig,
        provenance: ProvenanceConfig,
        asset: AssetConfig
    ):
        """Initialize CapitalGoodsConfig."""
        self.database = database
        self.calculation = calculation
        self.uncertainty = uncertainty
        self.compliance = compliance
        self.dqi = dqi
        self.api = api
        self.metrics = metrics
        self.provenance = provenance
        self.asset = asset

    @classmethod
    def from_env(cls) -> 'CapitalGoodsConfig':
        """
        Create configuration from environment variables.

        All environment variables use the GL_CG_* prefix for capital goods service.

        Returns:
            Configured CapitalGoodsConfig instance

        Example:
            >>> config = CapitalGoodsConfig.from_env()
        """
        with cls._lock:
            if cls._instance is None:
                # Database config
                database = DatabaseConfig(
                    host=os.getenv("GL_CG_DB_HOST", "localhost"),
                    port=int(os.getenv("GL_CG_DB_PORT", "5432")),
                    name=os.getenv("GL_CG_DB_NAME", "greenlang"),
                    user=os.getenv("GL_CG_DB_USER", "greenlang"),
                    password=os.getenv("GL_CG_DB_PASSWORD", ""),
                    pool_min=int(os.getenv("GL_CG_DB_POOL_MIN", "2")),
                    pool_max=int(os.getenv("GL_CG_DB_POOL_MAX", "10")),
                    ssl_mode=os.getenv("GL_CG_DB_SSL_MODE", "prefer"),
                    schema=os.getenv("GL_CG_DB_SCHEMA", "capital_goods_service"),
                    connection_timeout=int(os.getenv("GL_CG_DB_CONN_TIMEOUT", "30")),
                    command_timeout=int(os.getenv("GL_CG_DB_CMD_TIMEOUT", "60"))
                )

                # Calculation config
                calculation = CalculationConfig(
                    default_method=os.getenv("GL_CG_DEFAULT_METHOD", "hybrid"),
                    default_gwp_source=os.getenv("GL_CG_DEFAULT_GWP_SOURCE", "AR6"),
                    decimal_precision=int(os.getenv("GL_CG_DECIMAL_PRECISION", "10")),
                    rounding_mode=os.getenv("GL_CG_ROUNDING_MODE", "ROUND_HALF_UP"),
                    enable_spend_based=os.getenv("GL_CG_ENABLE_SPEND_BASED", "true").lower() == "true",
                    enable_average_data=os.getenv("GL_CG_ENABLE_AVERAGE_DATA", "true").lower() == "true",
                    enable_supplier_specific=os.getenv("GL_CG_ENABLE_SUPPLIER_SPECIFIC", "true").lower() == "true",
                    enable_hybrid=os.getenv("GL_CG_ENABLE_HYBRID", "true").lower() == "true",
                    eeio_fallback_digits=int(os.getenv("GL_CG_EEIO_FALLBACK_DIGITS", "2")),
                    cpi_base_year=int(os.getenv("GL_CG_CPI_BASE_YEAR", "2021")),
                    default_currency=os.getenv("GL_CG_DEFAULT_CURRENCY", "USD"),
                    enable_margin_removal=os.getenv("GL_CG_ENABLE_MARGIN_REMOVAL", "true").lower() == "true",
                    enable_transport_removal=os.getenv("GL_CG_ENABLE_TRANSPORT_REMOVAL", "true").lower() == "true",
                    default_margin_rate=Decimal(os.getenv("GL_CG_DEFAULT_MARGIN_RATE", "0.30")),
                    default_transport_rate=Decimal(os.getenv("GL_CG_DEFAULT_TRANSPORT_RATE", "0.05")),
                    capitalization_threshold_usd=Decimal(os.getenv("GL_CG_CAPITALIZATION_THRESHOLD_USD", "5000")),
                    default_capitalization_policy=os.getenv("GL_CG_DEFAULT_CAPITALIZATION_POLICY", "COMPANY_DEFINED"),
                    min_useful_life_years=int(os.getenv("GL_CG_MIN_USEFUL_LIFE_YEARS", "1")),
                    max_useful_life_years=int(os.getenv("GL_CG_MAX_USEFUL_LIFE_YEARS", "100")),
                    enable_asset_classification=os.getenv("GL_CG_ENABLE_ASSET_CLASSIFICATION", "true").lower() == "true",
                    enable_subcategory_resolution=os.getenv("GL_CG_ENABLE_SUBCATEGORY_RESOLUTION", "true").lower() == "true",
                    enable_useful_life_allocation=os.getenv("GL_CG_ENABLE_USEFUL_LIFE_ALLOCATION", "true").lower() == "true",
                    enable_first_year_convention=os.getenv("GL_CG_ENABLE_FIRST_YEAR_CONVENTION", "true").lower() == "true",
                    first_year_fraction=Decimal(os.getenv("GL_CG_FIRST_YEAR_FRACTION", "1.0")),
                    enable_capex_volatility_context=os.getenv("GL_CG_ENABLE_CAPEX_VOLATILITY_CONTEXT", "true").lower() == "true",
                    volatility_ratio_threshold=Decimal(os.getenv("GL_CG_VOLATILITY_RATIO_THRESHOLD", "2.0")),
                    rolling_average_years=int(os.getenv("GL_CG_ROLLING_AVERAGE_YEARS", "3")),
                    enable_leased_asset_exclusion=os.getenv("GL_CG_ENABLE_LEASED_ASSET_EXCLUSION", "true").lower() == "true",
                    require_purchase_date=os.getenv("GL_CG_REQUIRE_PURCHASE_DATE", "true").lower() == "true",
                    require_useful_life=os.getenv("GL_CG_REQUIRE_USEFUL_LIFE", "true").lower() == "true",
                    allow_category_fallback=os.getenv("GL_CG_ALLOW_CATEGORY_FALLBACK", "true").lower() == "true",
                    enable_double_counting_check=os.getenv("GL_CG_ENABLE_DOUBLE_COUNTING_CHECK", "true").lower() == "true",
                    enable_depreciation_exclusion=os.getenv("GL_CG_ENABLE_DEPRECIATION_EXCLUSION", "true").lower() == "true"
                )

                # Uncertainty config
                uncertainty = UncertaintyConfig(
                    enable=os.getenv("GL_CG_UNCERTAINTY_ENABLE", "true").lower() == "true",
                    method=os.getenv("GL_CG_UNCERTAINTY_METHOD", "monte_carlo"),
                    iterations=int(os.getenv("GL_CG_UNCERTAINTY_ITERATIONS", "5000")),
                    confidence_levels=[int(x) for x in os.getenv("GL_CG_UNCERTAINTY_CONFIDENCE_LEVELS", "90,95,99").split(",")],
                    seed=int(os.getenv("GL_CG_UNCERTAINTY_SEED", "42")) if os.getenv("GL_CG_UNCERTAINTY_SEED") else 42,
                    cv_spend_based=Decimal(os.getenv("GL_CG_CV_SPEND_BASED", "0.50")),
                    cv_average_data=Decimal(os.getenv("GL_CG_CV_AVERAGE_DATA", "0.35")),
                    cv_supplier_specific=Decimal(os.getenv("GL_CG_CV_SUPPLIER_SPECIFIC", "0.15")),
                    cv_hybrid=Decimal(os.getenv("GL_CG_CV_HYBRID", "0.25")),
                    cv_eeio_sector_match=Decimal(os.getenv("GL_CG_CV_EEIO_SECTOR_MATCH", "0.40")),
                    cv_eeio_fallback=Decimal(os.getenv("GL_CG_CV_EEIO_FALLBACK", "0.65")),
                    cv_inflation_adjustment=Decimal(os.getenv("GL_CG_CV_INFLATION_ADJUSTMENT", "0.08")),
                    cv_useful_life=Decimal(os.getenv("GL_CG_CV_USEFUL_LIFE", "0.20")),
                    enable_parameter_uncertainty=os.getenv("GL_CG_ENABLE_PARAMETER_UNCERTAINTY", "true").lower() == "true",
                    enable_model_uncertainty=os.getenv("GL_CG_ENABLE_MODEL_UNCERTAINTY", "true").lower() == "true",
                    enable_scenario_uncertainty=os.getenv("GL_CG_ENABLE_SCENARIO_UNCERTAINTY", "false").lower() == "true",
                    activity_distribution=os.getenv("GL_CG_ACTIVITY_DISTRIBUTION", "lognormal"),
                    ef_distribution=os.getenv("GL_CG_EF_DISTRIBUTION", "lognormal"),
                    enable_sensitivity_analysis=os.getenv("GL_CG_ENABLE_SENSITIVITY_ANALYSIS", "true").lower() == "true",
                    sensitivity_parameters=[
                        x.strip() for x in os.getenv(
                            "GL_CG_SENSITIVITY_PARAMETERS",
                            "spend_amount,emission_factor,useful_life,margin_rate,transport_rate"
                        ).split(",")
                    ]
                )

                # Compliance config
                compliance = ComplianceConfig(
                    enabled_frameworks=[
                        x.strip() for x in os.getenv(
                            "GL_CG_ENABLED_FRAMEWORKS",
                            "GHG_PROTOCOL_SCOPE3,ISO_14064_1,CSRD_ESRS_E1,CDP_SUPPLY_CHAIN,SBTI_SCOPE3,PCAF_INVESTMENTS,GRI_305"
                        ).split(",")
                    ],
                    ghg_protocol_version=os.getenv("GL_CG_GHG_PROTOCOL_VERSION", "2011"),
                    iso_14064_version=os.getenv("GL_CG_ISO_14064_VERSION", "2018"),
                    csrd_version=os.getenv("GL_CG_CSRD_VERSION", "2023"),
                    cdp_version=os.getenv("GL_CG_CDP_VERSION", "2024"),
                    sbti_version=os.getenv("GL_CG_SBTI_VERSION", "2.0"),
                    pcaf_version=os.getenv("GL_CG_PCAF_VERSION", "2022"),
                    gri_version=os.getenv("GL_CG_GRI_VERSION", "2021"),
                    enable_double_counting_check=os.getenv("GL_CG_COMPLIANCE_DOUBLE_COUNTING_CHECK", "true").lower() == "true",
                    require_base_year=os.getenv("GL_CG_REQUIRE_BASE_YEAR", "true").lower() == "true",
                    require_boundary_definition=os.getenv("GL_CG_REQUIRE_BOUNDARY_DEFINITION", "true").lower() == "true",
                    require_consolidation_approach=os.getenv("GL_CG_REQUIRE_CONSOLIDATION_APPROACH", "true").lower() == "true",
                    ghg_protocol_require_category_2=os.getenv("GL_CG_GHG_PROTOCOL_REQUIRE_CATEGORY_2", "true").lower() == "true",
                    ghg_protocol_require_useful_life=os.getenv("GL_CG_GHG_PROTOCOL_REQUIRE_USEFUL_LIFE", "true").lower() == "true",
                    ghg_protocol_allow_no_allocation=os.getenv("GL_CG_GHG_PROTOCOL_ALLOW_NO_ALLOCATION", "false").lower() == "true",
                    csrd_require_value_chain_mapping=os.getenv("GL_CG_CSRD_REQUIRE_VALUE_CHAIN_MAPPING", "true").lower() == "true",
                    csrd_require_double_materiality=os.getenv("GL_CG_CSRD_REQUIRE_DOUBLE_MATERIALITY", "true").lower() == "true",
                    sbti_require_upstream_coverage=os.getenv("GL_CG_SBTI_REQUIRE_UPSTREAM_COVERAGE", "true").lower() == "true",
                    sbti_minimum_coverage_percent=Decimal(os.getenv("GL_CG_SBTI_MINIMUM_COVERAGE_PERCENT", "67.0")),
                    pcaf_enable_data_quality_score=os.getenv("GL_CG_PCAF_ENABLE_DATA_QUALITY_SCORE", "true").lower() == "true",
                    pcaf_score_range=(1, 5),
                    materiality_threshold_tco2e=Decimal(os.getenv("GL_CG_MATERIALITY_THRESHOLD_TCO2E")) if os.getenv("GL_CG_MATERIALITY_THRESHOLD_TCO2E") else None,
                    reporting_currency=os.getenv("GL_CG_REPORTING_CURRENCY", "USD"),
                    enable_compliance_validation=os.getenv("GL_CG_ENABLE_COMPLIANCE_VALIDATION", "true").lower() == "true",
                    fail_on_non_compliance=os.getenv("GL_CG_FAIL_ON_NON_COMPLIANCE", "false").lower() == "true"
                )

                # DQI config
                dqi = DQIConfig(
                    enable=os.getenv("GL_CG_DQI_ENABLE", "true").lower() == "true",
                    weight_reliability=Decimal(os.getenv("GL_CG_DQI_WEIGHT_RELIABILITY", "0.25")),
                    weight_completeness=Decimal(os.getenv("GL_CG_DQI_WEIGHT_COMPLETENESS", "0.25")),
                    weight_temporal=Decimal(os.getenv("GL_CG_DQI_WEIGHT_TEMPORAL", "0.20")),
                    weight_geographical=Decimal(os.getenv("GL_CG_DQI_WEIGHT_GEOGRAPHICAL", "0.15")),
                    weight_technological=Decimal(os.getenv("GL_CG_DQI_WEIGHT_TECHNOLOGICAL", "0.15")),
                    quality_threshold_excellent=Decimal(os.getenv("GL_CG_DQI_THRESHOLD_EXCELLENT", "4.0")),
                    quality_threshold_good=Decimal(os.getenv("GL_CG_DQI_THRESHOLD_GOOD", "3.0")),
                    quality_threshold_fair=Decimal(os.getenv("GL_CG_DQI_THRESHOLD_FAIR", "2.0")),
                    quality_threshold_poor=Decimal(os.getenv("GL_CG_DQI_THRESHOLD_POOR", "1.0")),
                    minimum_acceptable_score=Decimal(os.getenv("GL_CG_DQI_MINIMUM_ACCEPTABLE_SCORE", "2.5")),
                    supplier_specific_score=Decimal(os.getenv("GL_CG_DQI_SUPPLIER_SPECIFIC_SCORE", "5.0")),
                    average_data_score=Decimal(os.getenv("GL_CG_DQI_AVERAGE_DATA_SCORE", "3.0")),
                    spend_based_score=Decimal(os.getenv("GL_CG_DQI_SPEND_BASED_SCORE", "2.0")),
                    temporal_max_age_years=int(os.getenv("GL_CG_DQI_TEMPORAL_MAX_AGE_YEARS", "5")),
                    temporal_score_current_year=Decimal(os.getenv("GL_CG_DQI_TEMPORAL_SCORE_CURRENT", "5.0")),
                    temporal_score_one_year=Decimal(os.getenv("GL_CG_DQI_TEMPORAL_SCORE_ONE_YEAR", "4.5")),
                    temporal_score_two_year=Decimal(os.getenv("GL_CG_DQI_TEMPORAL_SCORE_TWO_YEAR", "4.0")),
                    temporal_score_three_year=Decimal(os.getenv("GL_CG_DQI_TEMPORAL_SCORE_THREE_YEAR", "3.0")),
                    temporal_score_old=Decimal(os.getenv("GL_CG_DQI_TEMPORAL_SCORE_OLD", "2.0")),
                    geo_score_specific=Decimal(os.getenv("GL_CG_DQI_GEO_SCORE_SPECIFIC", "5.0")),
                    geo_score_regional=Decimal(os.getenv("GL_CG_DQI_GEO_SCORE_REGIONAL", "4.0")),
                    geo_score_continental=Decimal(os.getenv("GL_CG_DQI_GEO_SCORE_CONTINENTAL", "3.0")),
                    geo_score_global=Decimal(os.getenv("GL_CG_DQI_GEO_SCORE_GLOBAL", "2.0")),
                    tech_score_specific=Decimal(os.getenv("GL_CG_DQI_TECH_SCORE_SPECIFIC", "5.0")),
                    tech_score_sector_average=Decimal(os.getenv("GL_CG_DQI_TECH_SCORE_SECTOR_AVERAGE", "3.0")),
                    tech_score_generic=Decimal(os.getenv("GL_CG_DQI_TECH_SCORE_GENERIC", "2.0")),
                    enable_pcaf_mapping=os.getenv("GL_CG_DQI_ENABLE_PCAF_MAPPING", "true").lower() == "true",
                    dqi_to_pcaf_mapping={
                        "excellent": 1,
                        "good": 2,
                        "fair": 3,
                        "poor": 4,
                        "very_poor": 5
                    }
                )

                # API config
                api = APIConfig(
                    prefix=os.getenv("GL_CG_API_PREFIX", "/api/v1/capital-goods"),
                    max_batch_size=int(os.getenv("GL_CG_API_MAX_BATCH_SIZE", "500")),
                    pagination_default=int(os.getenv("GL_CG_API_PAGINATION_DEFAULT", "100")),
                    pagination_max=int(os.getenv("GL_CG_API_PAGINATION_MAX", "1000")),
                    rate_limit_requests=int(os.getenv("GL_CG_API_RATE_LIMIT_REQUESTS", "60")),
                    rate_limit_period=int(os.getenv("GL_CG_API_RATE_LIMIT_PERIOD", "60")),
                    cache_ttl=int(os.getenv("GL_CG_API_CACHE_TTL", "300")),
                    enable_cache=os.getenv("GL_CG_API_ENABLE_CACHE", "true").lower() == "true",
                    request_timeout=int(os.getenv("GL_CG_API_REQUEST_TIMEOUT", "30")),
                    calculation_timeout=int(os.getenv("GL_CG_API_CALCULATION_TIMEOUT", "120")),
                    batch_timeout=int(os.getenv("GL_CG_API_BATCH_TIMEOUT", "300")),
                    include_provenance=os.getenv("GL_CG_API_INCLUDE_PROVENANCE", "true").lower() == "true",
                    include_uncertainty=os.getenv("GL_CG_API_INCLUDE_UNCERTAINTY", "true").lower() == "true",
                    include_dqi=os.getenv("GL_CG_API_INCLUDE_DQI", "true").lower() == "true",
                    include_compliance=os.getenv("GL_CG_API_INCLUDE_COMPLIANCE", "true").lower() == "true",
                    return_partial_results=os.getenv("GL_CG_API_RETURN_PARTIAL_RESULTS", "false").lower() == "true",
                    max_error_detail_length=int(os.getenv("GL_CG_API_MAX_ERROR_DETAIL_LENGTH", "500"))
                )

                # Metrics config
                metrics = MetricsConfig(
                    enable=os.getenv("GL_CG_METRICS_ENABLE", "true").lower() == "true",
                    prefix=os.getenv("GL_CG_METRICS_PREFIX", "gl_cg"),
                    namespace=os.getenv("GL_CG_METRICS_NAMESPACE", "greenlang"),
                    subsystem=os.getenv("GL_CG_METRICS_SUBSYSTEM", "capital_goods"),
                    histogram_buckets=[float(x) for x in os.getenv("GL_CG_METRICS_HISTOGRAM_BUCKETS", "0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0").split(",")],
                    enable_calculation_counter=os.getenv("GL_CG_METRICS_ENABLE_CALCULATION_COUNTER", "true").lower() == "true",
                    enable_error_counter=os.getenv("GL_CG_METRICS_ENABLE_ERROR_COUNTER", "true").lower() == "true",
                    enable_validation_counter=os.getenv("GL_CG_METRICS_ENABLE_VALIDATION_COUNTER", "true").lower() == "true",
                    enable_active_calculations=os.getenv("GL_CG_METRICS_ENABLE_ACTIVE_CALCULATIONS", "true").lower() == "true",
                    enable_cache_size=os.getenv("GL_CG_METRICS_ENABLE_CACHE_SIZE", "true").lower() == "true",
                    enable_processing_time=os.getenv("GL_CG_METRICS_ENABLE_PROCESSING_TIME", "true").lower() == "true",
                    enable_emission_distribution=os.getenv("GL_CG_METRICS_ENABLE_EMISSION_DISTRIBUTION", "true").lower() == "true",
                    enable_dqi_distribution=os.getenv("GL_CG_METRICS_ENABLE_DQI_DISTRIBUTION", "true").lower() == "true",
                    default_labels=[x.strip() for x in os.getenv("GL_CG_METRICS_DEFAULT_LABELS", "method,asset_category,framework,status").split(",")]
                )

                # Provenance config
                provenance = ProvenanceConfig(
                    enable=os.getenv("GL_CG_PROVENANCE_ENABLE", "true").lower() == "true",
                    hash_algorithm=os.getenv("GL_CG_PROVENANCE_HASH_ALGORITHM", "sha256"),
                    chain_hashing=os.getenv("GL_CG_PROVENANCE_CHAIN_HASHING", "true").lower() == "true",
                    stage_names=[
                        x.strip() for x in os.getenv(
                            "GL_CG_PROVENANCE_STAGE_NAMES",
                            "input_validation,capitalization_check,asset_classification,useful_life_determination,method_selection,spend_normalization,margin_removal,transport_removal,eeio_matching,emission_factor_lookup,emission_calculation,useful_life_allocation,uncertainty_quantification,dqi_calculation,compliance_validation,aggregation,output_generation"
                        ).split(",")
                    ],
                    store_intermediate_results=os.getenv("GL_CG_PROVENANCE_STORE_INTERMEDIATE_RESULTS", "true").lower() == "true",
                    compress_provenance=os.getenv("GL_CG_PROVENANCE_COMPRESS", "true").lower() == "true",
                    retention_days=int(os.getenv("GL_CG_PROVENANCE_RETENTION_DAYS", "2555")),
                    auto_cleanup=os.getenv("GL_CG_PROVENANCE_AUTO_CLEANUP", "true").lower() == "true",
                    track_user_actions=os.getenv("GL_CG_PROVENANCE_TRACK_USER_ACTIONS", "true").lower() == "true",
                    track_data_lineage=os.getenv("GL_CG_PROVENANCE_TRACK_DATA_LINEAGE", "true").lower() == "true",
                    track_assumption_changes=os.getenv("GL_CG_PROVENANCE_TRACK_ASSUMPTION_CHANGES", "true").lower() == "true"
                )

                # Asset config
                asset = AssetConfig(
                    enable_asset_classification=os.getenv("GL_CG_ASSET_ENABLE_CLASSIFICATION", "true").lower() == "true",
                    enable_subcategory_resolution=os.getenv("GL_CG_ASSET_ENABLE_SUBCATEGORY_RESOLUTION", "true").lower() == "true",
                    default_useful_life=int(os.getenv("GL_CG_ASSET_DEFAULT_USEFUL_LIFE", "10")),
                    useful_life_buildings=int(os.getenv("GL_CG_ASSET_USEFUL_LIFE_BUILDINGS", "40")),
                    useful_life_machinery=int(os.getenv("GL_CG_ASSET_USEFUL_LIFE_MACHINERY", "15")),
                    useful_life_vehicles=int(os.getenv("GL_CG_ASSET_USEFUL_LIFE_VEHICLES", "8")),
                    useful_life_it_equipment=int(os.getenv("GL_CG_ASSET_USEFUL_LIFE_IT_EQUIPMENT", "5")),
                    useful_life_furniture=int(os.getenv("GL_CG_ASSET_USEFUL_LIFE_FURNITURE", "10")),
                    useful_life_infrastructure=int(os.getenv("GL_CG_ASSET_USEFUL_LIFE_INFRASTRUCTURE", "50")),
                    useful_life_tools=int(os.getenv("GL_CG_ASSET_USEFUL_LIFE_TOOLS", "7")),
                    useful_life_other=int(os.getenv("GL_CG_ASSET_USEFUL_LIFE_OTHER", "10")),
                    threshold_buildings=Decimal(os.getenv("GL_CG_ASSET_THRESHOLD_BUILDINGS", "50000")),
                    threshold_machinery=Decimal(os.getenv("GL_CG_ASSET_THRESHOLD_MACHINERY", "10000")),
                    threshold_vehicles=Decimal(os.getenv("GL_CG_ASSET_THRESHOLD_VEHICLES", "5000")),
                    threshold_it_equipment=Decimal(os.getenv("GL_CG_ASSET_THRESHOLD_IT_EQUIPMENT", "1000")),
                    threshold_furniture=Decimal(os.getenv("GL_CG_ASSET_THRESHOLD_FURNITURE", "2500")),
                    threshold_infrastructure=Decimal(os.getenv("GL_CG_ASSET_THRESHOLD_INFRASTRUCTURE", "100000")),
                    threshold_tools=Decimal(os.getenv("GL_CG_ASSET_THRESHOLD_TOOLS", "500")),
                    threshold_other=Decimal(os.getenv("GL_CG_ASSET_THRESHOLD_OTHER", "5000")),
                    enable_leased_asset_exclusion=os.getenv("GL_CG_ASSET_ENABLE_LEASED_ASSET_EXCLUSION", "true").lower() == "true",
                    leased_asset_categories=[x.strip() for x in os.getenv("GL_CG_ASSET_LEASED_ASSET_CATEGORIES", "finance_lease,operating_lease,rental").split(",")],
                    depreciation_method_straight_line=os.getenv("GL_CG_ASSET_DEPRECIATION_STRAIGHT_LINE", "true").lower() == "true",
                    depreciation_method_declining_balance=os.getenv("GL_CG_ASSET_DEPRECIATION_DECLINING_BALANCE", "true").lower() == "true",
                    depreciation_method_units_of_production=os.getenv("GL_CG_ASSET_DEPRECIATION_UNITS_OF_PRODUCTION", "true").lower() == "true",
                    require_asset_id=os.getenv("GL_CG_ASSET_REQUIRE_ASSET_ID", "true").lower() == "true",
                    require_purchase_order=os.getenv("GL_CG_ASSET_REQUIRE_PURCHASE_ORDER", "false").lower() == "true",
                    require_supplier_info=os.getenv("GL_CG_ASSET_REQUIRE_SUPPLIER_INFO", "false").lower() == "true",
                    max_useful_life=int(os.getenv("GL_CG_ASSET_MAX_USEFUL_LIFE", "100")),
                    min_useful_life=int(os.getenv("GL_CG_ASSET_MIN_USEFUL_LIFE", "1"))
                )

                cls._instance = cls(
                    database=database,
                    calculation=calculation,
                    uncertainty=uncertainty,
                    compliance=compliance,
                    dqi=dqi,
                    api=api,
                    metrics=metrics,
                    provenance=provenance,
                    asset=asset
                )

                logger.info("CapitalGoodsConfig initialized from environment")

            return cls._instance

    def validate(self) -> None:
        """
        Validate all configuration settings.

        Raises:
            ValueError: If any configuration is invalid

        Example:
            >>> config = CapitalGoodsConfig.from_env()
            >>> config.validate()
        """
        errors = []

        # Database validation
        if not self.database.host:
            errors.append("Database host is required")
        if self.database.port <= 0 or self.database.port > 65535:
            errors.append(f"Invalid database port: {self.database.port}")
        if self.database.pool_min < 0:
            errors.append(f"Pool min must be >= 0: {self.database.pool_min}")
        if self.database.pool_max < self.database.pool_min:
            errors.append(f"Pool max ({self.database.pool_max}) must be >= pool min ({self.database.pool_min})")

        # Calculation validation
        valid_methods = [m.value for m in CalculationMethod]
        if self.calculation.default_method not in valid_methods:
            errors.append(f"Invalid calculation method: {self.calculation.default_method}")

        valid_gwp_sources = [s.value for s in GWPSource]
        if self.calculation.default_gwp_source not in valid_gwp_sources:
            errors.append(f"Invalid GWP source: {self.calculation.default_gwp_source}")

        if self.calculation.decimal_precision < 1 or self.calculation.decimal_precision > 28:
            errors.append(f"Decimal precision must be 1-28: {self.calculation.decimal_precision}")

        if self.calculation.eeio_fallback_digits < 1 or self.calculation.eeio_fallback_digits > 6:
            errors.append(f"EEIO fallback digits must be 1-6: {self.calculation.eeio_fallback_digits}")

        if self.calculation.default_margin_rate < 0 or self.calculation.default_margin_rate > 1:
            errors.append(f"Margin rate must be 0-1: {self.calculation.default_margin_rate}")

        if self.calculation.default_transport_rate < 0 or self.calculation.default_transport_rate > 1:
            errors.append(f"Transport rate must be 0-1: {self.calculation.default_transport_rate}")

        if self.calculation.capitalization_threshold_usd < 0:
            errors.append(f"Capitalization threshold must be >= 0: {self.calculation.capitalization_threshold_usd}")

        if self.calculation.min_useful_life_years < 1:
            errors.append(f"Min useful life must be >= 1: {self.calculation.min_useful_life_years}")

        if self.calculation.max_useful_life_years < self.calculation.min_useful_life_years:
            errors.append(f"Max useful life ({self.calculation.max_useful_life_years}) must be >= min ({self.calculation.min_useful_life_years})")

        if self.calculation.volatility_ratio_threshold < 1:
            errors.append(f"Volatility ratio threshold must be >= 1: {self.calculation.volatility_ratio_threshold}")

        if self.calculation.rolling_average_years < 1:
            errors.append(f"Rolling average years must be >= 1: {self.calculation.rolling_average_years}")

        # Uncertainty validation
        if self.uncertainty.enable:
            valid_uncertainty_methods = [m.value for m in UncertaintyMethod]
            if self.uncertainty.method not in valid_uncertainty_methods:
                errors.append(f"Invalid uncertainty method: {self.uncertainty.method}")

            if self.uncertainty.iterations < 100:
                errors.append(f"Uncertainty iterations must be >= 100: {self.uncertainty.iterations}")

            if not self.uncertainty.confidence_levels:
                errors.append("At least one confidence level is required")

            for cl in self.uncertainty.confidence_levels:
                if cl < 1 or cl > 99:
                    errors.append(f"Confidence level must be 1-99: {cl}")

            cv_fields = [
                ("cv_spend_based", self.uncertainty.cv_spend_based),
                ("cv_average_data", self.uncertainty.cv_average_data),
                ("cv_supplier_specific", self.uncertainty.cv_supplier_specific),
                ("cv_hybrid", self.uncertainty.cv_hybrid),
                ("cv_eeio_sector_match", self.uncertainty.cv_eeio_sector_match),
                ("cv_eeio_fallback", self.uncertainty.cv_eeio_fallback),
                ("cv_inflation_adjustment", self.uncertainty.cv_inflation_adjustment),
                ("cv_useful_life", self.uncertainty.cv_useful_life)
            ]

            for name, value in cv_fields:
                if value < 0:
                    errors.append(f"{name} must be >= 0: {value}")

        # Compliance validation
        valid_frameworks = [f.value for f in RegulatoryFramework]
        for framework in self.compliance.enabled_frameworks:
            if framework not in valid_frameworks:
                errors.append(f"Invalid framework: {framework}")

        if self.compliance.sbti_minimum_coverage_percent < 0 or self.compliance.sbti_minimum_coverage_percent > 100:
            errors.append(f"SBTi minimum coverage must be 0-100: {self.compliance.sbti_minimum_coverage_percent}")

        # DQI validation
        if self.dqi.enable:
            weight_sum = (
                self.dqi.weight_reliability +
                self.dqi.weight_completeness +
                self.dqi.weight_temporal +
                self.dqi.weight_geographical +
                self.dqi.weight_technological
            )

            if abs(weight_sum - Decimal("1.0")) > Decimal("0.001"):
                errors.append(f"DQI weights must sum to 1.0, got {weight_sum}")

            for weight_name in ["reliability", "completeness", "temporal", "geographical", "technological"]:
                weight = getattr(self.dqi, f"weight_{weight_name}")
                if weight < 0 or weight > 1:
                    errors.append(f"DQI weight_{weight_name} must be 0-1: {weight}")

            if self.dqi.minimum_acceptable_score < 1 or self.dqi.minimum_acceptable_score > 5:
                errors.append(f"Minimum acceptable score must be 1-5: {self.dqi.minimum_acceptable_score}")

            if self.dqi.temporal_max_age_years < 0:
                errors.append(f"Temporal max age must be >= 0: {self.dqi.temporal_max_age_years}")

        # API validation
        if self.api.max_batch_size < 1:
            errors.append(f"Max batch size must be >= 1: {self.api.max_batch_size}")

        if self.api.pagination_default < 1:
            errors.append(f"Pagination default must be >= 1: {self.api.pagination_default}")

        if self.api.pagination_max < self.api.pagination_default:
            errors.append(f"Pagination max ({self.api.pagination_max}) must be >= default ({self.api.pagination_default})")

        if self.api.rate_limit_requests < 1:
            errors.append(f"Rate limit requests must be >= 1: {self.api.rate_limit_requests}")

        if self.api.rate_limit_period < 1:
            errors.append(f"Rate limit period must be >= 1: {self.api.rate_limit_period}")

        if self.api.cache_ttl < 0:
            errors.append(f"Cache TTL must be >= 0: {self.api.cache_ttl}")

        # Metrics validation
        if self.metrics.enable:
            if not self.metrics.prefix:
                errors.append("Metrics prefix is required")

            if not self.metrics.histogram_buckets:
                errors.append("Histogram buckets are required")

            if len(self.metrics.histogram_buckets) < 2:
                errors.append("At least 2 histogram buckets are required")

            for i in range(len(self.metrics.histogram_buckets) - 1):
                if self.metrics.histogram_buckets[i] >= self.metrics.histogram_buckets[i + 1]:
                    errors.append("Histogram buckets must be in ascending order")
                    break

        # Provenance validation
        if self.provenance.enable:
            if self.provenance.hash_algorithm not in ["sha256", "sha512", "blake2b"]:
                errors.append(f"Invalid hash algorithm: {self.provenance.hash_algorithm}")

            if not self.provenance.stage_names:
                errors.append("Provenance stage names are required")

            if self.provenance.retention_days < 1:
                errors.append(f"Retention days must be >= 1: {self.provenance.retention_days}")

        # Asset validation
        if self.asset.min_useful_life < 1:
            errors.append(f"Asset min useful life must be >= 1: {self.asset.min_useful_life}")

        if self.asset.max_useful_life < self.asset.min_useful_life:
            errors.append(f"Asset max useful life ({self.asset.max_useful_life}) must be >= min ({self.asset.min_useful_life})")

        useful_life_fields = [
            ("buildings", self.asset.useful_life_buildings),
            ("machinery", self.asset.useful_life_machinery),
            ("vehicles", self.asset.useful_life_vehicles),
            ("it_equipment", self.asset.useful_life_it_equipment),
            ("furniture", self.asset.useful_life_furniture),
            ("infrastructure", self.asset.useful_life_infrastructure),
            ("tools", self.asset.useful_life_tools),
            ("other", self.asset.useful_life_other)
        ]

        for name, value in useful_life_fields:
            if value < self.asset.min_useful_life or value > self.asset.max_useful_life:
                errors.append(f"Useful life for {name} ({value}) must be between {self.asset.min_useful_life} and {self.asset.max_useful_life}")

        threshold_fields = [
            ("buildings", self.asset.threshold_buildings),
            ("machinery", self.asset.threshold_machinery),
            ("vehicles", self.asset.threshold_vehicles),
            ("it_equipment", self.asset.threshold_it_equipment),
            ("furniture", self.asset.threshold_furniture),
            ("infrastructure", self.asset.threshold_infrastructure),
            ("tools", self.asset.threshold_tools),
            ("other", self.asset.threshold_other)
        ]

        for name, value in threshold_fields:
            if value < 0:
                errors.append(f"Threshold for {name} must be >= 0: {value}")

        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        logger.info("Configuration validation passed")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration

        Example:
            >>> config = CapitalGoodsConfig.from_env()
            >>> config_dict = config.to_dict()
        """
        return {
            "database": asdict(self.database),
            "calculation": asdict(self.calculation),
            "uncertainty": asdict(self.uncertainty),
            "compliance": asdict(self.compliance),
            "dqi": asdict(self.dqi),
            "api": asdict(self.api),
            "metrics": asdict(self.metrics),
            "provenance": asdict(self.provenance),
            "asset": asdict(self.asset)
        }

    @classmethod
    def reset(cls) -> None:
        """
        Reset singleton instance.

        Use this for testing or when configuration needs to be reloaded.

        Example:
            >>> CapitalGoodsConfig.reset()
            >>> config = CapitalGoodsConfig.from_env()
        """
        with cls._lock:
            cls._instance = None
            logger.info("CapitalGoodsConfig singleton reset")


# ============================================================================
# Helper Functions
# ============================================================================


def get_config() -> CapitalGoodsConfig:
    """
    Get or create CapitalGoodsConfig singleton instance.

    Returns:
        CapitalGoodsConfig singleton instance

    Example:
        >>> config = get_config()
        >>> print(config.calculation.default_method)
        'hybrid'
    """
    return CapitalGoodsConfig.from_env()
