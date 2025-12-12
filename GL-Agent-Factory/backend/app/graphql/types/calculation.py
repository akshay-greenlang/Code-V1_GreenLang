"""
Calculation Result GraphQL Types

Defines GraphQL types for calculation results, provenance tracking,
and data quality scoring for Process Heat agent calculations.

Features:
- Full provenance chain tracking
- Emission factor references
- Data quality scoring
- Unit conversion support
- Validation results

Example:
    query {
        calculation(id: "calc-123") {
            id
            status
            result {
                value
                unit
                confidence
            }
            provenance {
                inputHash
                outputHash
                chainHash
            }
        }
    }
"""

import strawberry
from strawberry.scalars import JSON
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


# =============================================================================
# Custom Scalars
# =============================================================================


@strawberry.scalar(
    description="Date and time in ISO 8601 format",
    serialize=lambda v: v.isoformat() if v else None,
    parse_value=lambda v: datetime.fromisoformat(v) if v else None,
)
class DateTime:
    """Custom DateTime scalar for ISO 8601 format."""
    pass


# =============================================================================
# Enums
# =============================================================================


@strawberry.enum
class CalculationStatusEnum(Enum):
    """Calculation execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    VALIDATING = "validating"


@strawberry.enum
class DataQualityTier(Enum):
    """Data quality tier based on GHG Protocol."""

    TIER_1 = "tier_1"  # Highest quality - direct measurements
    TIER_2 = "tier_2"  # High quality - facility-specific data
    TIER_3 = "tier_3"  # Medium quality - industry averages
    TIER_4 = "tier_4"  # Low quality - estimates/proxies


@strawberry.enum
class UncertaintyType(Enum):
    """Type of uncertainty in calculations."""

    MEASUREMENT = "measurement"
    MODEL = "model"
    EMISSION_FACTOR = "emission_factor"
    ACTIVITY_DATA = "activity_data"
    COMBINED = "combined"


@strawberry.enum
class MethodologyType(Enum):
    """Calculation methodology types."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    ISO_50001 = "iso_50001"
    CBAM = "cbam"
    EU_ETS = "eu_ets"
    EPA_MANDATORY = "epa_mandatory"
    CUSTOM = "custom"


# =============================================================================
# Input Types
# =============================================================================


@strawberry.input
class CalculationInputType:
    """Input for running a calculation."""

    agent_id: str = strawberry.field(
        description="Agent ID to run (e.g., GL-022)"
    )
    parameters: JSON = strawberry.field(
        description="Calculation input parameters"
    )
    methodology: Optional[str] = strawberry.field(
        default="ghg_protocol",
        description="Calculation methodology to use"
    )
    reporting_period_start: Optional[str] = strawberry.field(
        default=None,
        description="Start of reporting period (ISO date)"
    )
    reporting_period_end: Optional[str] = strawberry.field(
        default=None,
        description="End of reporting period (ISO date)"
    )
    include_uncertainty: Optional[bool] = strawberry.field(
        default=True,
        description="Include uncertainty analysis"
    )
    quality_tier: Optional[str] = strawberry.field(
        default="tier_2",
        description="Target data quality tier"
    )


# =============================================================================
# Object Types
# =============================================================================


@strawberry.type
class EmissionFactorType:
    """Emission factor used in calculations."""

    id: str = strawberry.field(description="Emission factor ID")
    name: str = strawberry.field(description="Factor name")
    value: float = strawberry.field(description="Factor value")
    unit: str = strawberry.field(description="Unit of measurement")
    source: str = strawberry.field(description="Data source (e.g., IPCC, EPA)")
    source_url: Optional[str] = strawberry.field(
        default=None,
        description="URL to source documentation"
    )
    valid_from: DateTime = strawberry.field(description="Validity start date")
    valid_to: Optional[DateTime] = strawberry.field(
        default=None,
        description="Validity end date"
    )
    region: Optional[str] = strawberry.field(
        default=None,
        description="Geographic region"
    )
    sector: Optional[str] = strawberry.field(
        default=None,
        description="Industry sector"
    )
    methodology: str = strawberry.field(description="Calculation methodology")
    uncertainty_percent: Optional[float] = strawberry.field(
        default=None,
        description="Uncertainty as percentage"
    )
    quality_tier: DataQualityTier = strawberry.field(
        description="Data quality tier"
    )


@strawberry.type
class UnitConversionType:
    """Unit conversion applied in calculation."""

    from_value: float = strawberry.field(description="Original value")
    from_unit: str = strawberry.field(description="Original unit")
    to_value: float = strawberry.field(description="Converted value")
    to_unit: str = strawberry.field(description="Target unit")
    conversion_factor: float = strawberry.field(description="Conversion factor used")
    source: str = strawberry.field(
        default="QUDT",
        description="Conversion source (e.g., QUDT, custom)"
    )


@strawberry.type
class UncertaintyAnalysis:
    """Uncertainty analysis for a calculation."""

    type: UncertaintyType = strawberry.field(description="Type of uncertainty")
    lower_bound: float = strawberry.field(description="Lower bound (95% CI)")
    upper_bound: float = strawberry.field(description="Upper bound (95% CI)")
    standard_deviation: float = strawberry.field(description="Standard deviation")
    coefficient_of_variation: float = strawberry.field(
        description="Coefficient of variation"
    )
    confidence_level: float = strawberry.field(
        default=0.95,
        description="Confidence level"
    )
    method: str = strawberry.field(
        default="monte_carlo",
        description="Uncertainty calculation method"
    )


@strawberry.type
class ProvenanceType:
    """Calculation provenance for audit trail."""

    input_hash: str = strawberry.field(
        description="SHA-256 hash of input data"
    )
    output_hash: str = strawberry.field(
        description="SHA-256 hash of output data"
    )
    chain_hash: str = strawberry.field(
        description="Full provenance chain hash"
    )
    agent_version: str = strawberry.field(
        description="Agent version used"
    )
    calculation_timestamp: DateTime = strawberry.field(
        description="When calculation was performed"
    )
    emission_factor_ids: List[str] = strawberry.field(
        description="IDs of emission factors used"
    )
    methodology: str = strawberry.field(
        description="Calculation methodology"
    )
    regulatory_framework: Optional[str] = strawberry.field(
        default=None,
        description="Applicable regulatory framework"
    )
    audit_trail: List[str] = strawberry.field(
        default_factory=list,
        description="Audit trail entries"
    )
    parent_calculation_id: Optional[str] = strawberry.field(
        default=None,
        description="Parent calculation if derived"
    )
    is_verified: bool = strawberry.field(
        default=False,
        description="Whether provenance has been verified"
    )


@strawberry.type
class ValidationResultType:
    """Validation result for input or output data."""

    is_valid: bool = strawberry.field(description="Overall validation status")
    errors: List[str] = strawberry.field(
        default_factory=list,
        description="Validation errors"
    )
    warnings: List[str] = strawberry.field(
        default_factory=list,
        description="Validation warnings"
    )
    field_validations: JSON = strawberry.field(
        default_factory=dict,
        description="Per-field validation results"
    )
    validated_at: DateTime = strawberry.field(description="Validation timestamp")
    validator_version: str = strawberry.field(
        default="1.0.0",
        description="Validator version"
    )


@strawberry.type
class QualityScoreType:
    """Data quality score for calculation."""

    overall_score: float = strawberry.field(
        description="Overall quality score (0-100)"
    )
    tier: DataQualityTier = strawberry.field(
        description="Quality tier"
    )
    completeness: float = strawberry.field(
        description="Data completeness score (0-100)"
    )
    accuracy: float = strawberry.field(
        description="Data accuracy score (0-100)"
    )
    consistency: float = strawberry.field(
        description="Data consistency score (0-100)"
    )
    timeliness: float = strawberry.field(
        description="Data timeliness score (0-100)"
    )
    representativeness: float = strawberry.field(
        description="Data representativeness score (0-100)"
    )
    recommendations: List[str] = strawberry.field(
        default_factory=list,
        description="Recommendations for improving quality"
    )


@strawberry.type
class CalculationOutputType:
    """Output of a calculation."""

    value: float = strawberry.field(description="Primary result value")
    unit: str = strawberry.field(description="Unit of measurement")
    display_value: str = strawberry.field(description="Formatted display value")
    secondary_values: JSON = strawberry.field(
        default_factory=dict,
        description="Additional output values"
    )
    breakdown: JSON = strawberry.field(
        default_factory=dict,
        description="Breakdown of calculation components"
    )
    comparisons: JSON = strawberry.field(
        default_factory=dict,
        description="Comparison with benchmarks"
    )

    @strawberry.field(description="Get formatted result with unit")
    def formatted(self, precision: int = 2) -> str:
        """Get formatted result string."""
        return f"{self.value:.{precision}f} {self.unit}"


@strawberry.type
class CalculationResultType:
    """Complete calculation result with provenance."""

    # Identification
    id: str = strawberry.field(description="Calculation ID")
    execution_id: str = strawberry.field(description="Execution ID")
    agent_id: str = strawberry.field(description="Agent that performed calculation")

    # Status
    status: CalculationStatusEnum = strawberry.field(description="Calculation status")
    progress_percent: int = strawberry.field(
        default=0,
        description="Progress percentage (0-100)"
    )

    # Timing
    started_at: DateTime = strawberry.field(description="Start timestamp")
    completed_at: Optional[DateTime] = strawberry.field(
        default=None,
        description="Completion timestamp"
    )
    duration_ms: Optional[float] = strawberry.field(
        default=None,
        description="Duration in milliseconds"
    )

    # Results
    result: Optional[CalculationOutputType] = strawberry.field(
        default=None,
        description="Calculation output"
    )
    error_message: Optional[str] = strawberry.field(
        default=None,
        description="Error message if failed"
    )
    error_code: Optional[str] = strawberry.field(
        default=None,
        description="Error code if failed"
    )

    # Quality & Validation
    confidence_score: float = strawberry.field(
        default=0.0,
        description="Confidence score (0-1)"
    )
    quality_score: Optional[QualityScoreType] = strawberry.field(
        default=None,
        description="Data quality assessment"
    )
    validation: Optional[ValidationResultType] = strawberry.field(
        default=None,
        description="Validation results"
    )

    # Provenance
    provenance: Optional[ProvenanceType] = strawberry.field(
        default=None,
        description="Calculation provenance"
    )

    # References
    emission_factors: List[EmissionFactorType] = strawberry.field(
        default_factory=list,
        description="Emission factors used"
    )
    unit_conversions: List[UnitConversionType] = strawberry.field(
        default_factory=list,
        description="Unit conversions applied"
    )

    # Uncertainty
    uncertainty: Optional[UncertaintyAnalysis] = strawberry.field(
        default=None,
        description="Uncertainty analysis"
    )

    # Metadata
    methodology: str = strawberry.field(
        default="ghg_protocol",
        description="Calculation methodology"
    )
    inputs: JSON = strawberry.field(
        default_factory=dict,
        description="Original input data"
    )
    metadata: JSON = strawberry.field(
        default_factory=dict,
        description="Additional metadata"
    )

    # Multi-tenancy
    tenant_id: str = strawberry.field(description="Tenant ID")
    created_by: Optional[str] = strawberry.field(
        default=None,
        description="User who initiated calculation"
    )

    @strawberry.field(description="Is calculation complete")
    def is_complete(self) -> bool:
        """Check if calculation is complete."""
        return self.status in [
            CalculationStatusEnum.COMPLETED,
            CalculationStatusEnum.FAILED,
            CalculationStatusEnum.CANCELLED,
            CalculationStatusEnum.TIMEOUT
        ]

    @strawberry.field(description="Is calculation successful")
    def is_successful(self) -> bool:
        """Check if calculation completed successfully."""
        return self.status == CalculationStatusEnum.COMPLETED and self.result is not None


# =============================================================================
# Connection Types
# =============================================================================


@strawberry.type
class CalculationEdge:
    """Edge in calculation connection."""

    cursor: str = strawberry.field(description="Cursor for this item")
    node: CalculationResultType = strawberry.field(description="The calculation result")


@strawberry.type
class CalculationConnection:
    """Paginated connection of calculation results."""

    edges: List[CalculationEdge] = strawberry.field(description="List of edges")
    page_info: "PageInfo" = strawberry.field(description="Pagination info")


@strawberry.type
class PageInfo:
    """Pagination information."""

    has_next_page: bool = strawberry.field(description="More pages available after")
    has_previous_page: bool = strawberry.field(description="More pages available before")
    start_cursor: Optional[str] = strawberry.field(description="Cursor of first item")
    end_cursor: Optional[str] = strawberry.field(description="Cursor of last item")
    total_count: int = strawberry.field(description="Total number of items")


# =============================================================================
# Batch Types
# =============================================================================


@strawberry.type
class BatchCalculationResultType:
    """Result of a batch calculation."""

    batch_id: str = strawberry.field(description="Batch ID")
    total_calculations: int = strawberry.field(description="Total calculations in batch")
    completed_calculations: int = strawberry.field(description="Completed calculations")
    failed_calculations: int = strawberry.field(description="Failed calculations")
    status: CalculationStatusEnum = strawberry.field(description="Overall batch status")
    results: List[CalculationResultType] = strawberry.field(
        description="Individual calculation results"
    )
    started_at: DateTime = strawberry.field(description="Batch start time")
    completed_at: Optional[DateTime] = strawberry.field(
        default=None,
        description="Batch completion time"
    )
    total_duration_ms: Optional[float] = strawberry.field(
        default=None,
        description="Total batch duration"
    )
