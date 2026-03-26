"""
PACK-049 GHG Multi-Site Management Pack - Site Data Collection Engine
====================================================================

Manages the collection of GHG activity data from individual sites
within a multi-site organisation. Provides structured collection rounds,
template generation, data submission, validation, approval workflows,
and missing-data estimation capabilities.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 6): Identifying and
      calculating GHG emissions requires systematic data collection
      from all included facilities.
    - ISO 14064-1:2018 (Clause 5.2): Organisation shall quantify
      GHG emissions using collected activity data.
    - GHG Protocol Scope 3 (Chapter 7): Data collection requirements
      for value chain emissions.
    - ESRS E1-6: Gross Scopes 1, 2, 3 require documented data
      collection processes with quality controls.

Capabilities:
    - Create time-bounded collection rounds with deadlines
    - Generate site-specific data collection templates
    - Submit and validate activity data per site
    - Range, year-on-year variance, unit, and completeness validation
    - Approval and rejection workflow for submissions
    - Missing data estimation (extrapolation, proxy, average methods)
    - Collection round status and progress tracking

Zero-Hallucination:
    - All validations use deterministic rules with configurable params
    - Estimation methods use documented, auditable formulas
    - All calculations use Decimal arithmetic
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-049 GHG Multi-Site Management
Engine:  2 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash, excluding volatile fields."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("created_at", "updated_at", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert any value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> Decimal:
    """Round a value to two decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _round4(value: Any) -> Decimal:
    """Round a value to four decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataSource(str, Enum):
    """Source of the data entry."""
    MANUAL = "MANUAL"
    METERED = "METERED"
    INVOICE = "INVOICE"
    ERP = "ERP"
    IOT = "IOT"
    ESTIMATED = "ESTIMATED"
    THIRD_PARTY = "THIRD_PARTY"


class SubmissionStatus(str, Enum):
    """Status of a site data submission."""
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVISION_REQUIRED = "REVISION_REQUIRED"


class RoundStatus(str, Enum):
    """Status of a collection round."""
    PLANNED = "PLANNED"
    OPEN = "OPEN"
    EXTENDED = "EXTENDED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FINALISED = "FINALISED"


class PeriodType(str, Enum):
    """Reporting period granularity."""
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    ANNUAL = "ANNUAL"


class EstimationMethod(str, Enum):
    """Methods for estimating missing data."""
    EXTRAPOLATION = "EXTRAPOLATION"
    PROXY_DATA = "PROXY_DATA"
    SECTOR_AVERAGE = "SECTOR_AVERAGE"
    PRIOR_YEAR = "PRIOR_YEAR"
    LINEAR_INTERPOLATION = "LINEAR_INTERPOLATION"
    INTENSITY_BASED = "INTENSITY_BASED"


class ValidationSeverity(str, Enum):
    """Severity of a validation finding."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class RuleType(str, Enum):
    """Type of validation rule."""
    RANGE_CHECK = "RANGE_CHECK"
    YOY_VARIANCE = "YOY_VARIANCE"
    UNIT_CHECK = "UNIT_CHECK"
    COMPLETENESS = "COMPLETENESS"
    CROSS_FIELD = "CROSS_FIELD"
    CUSTOM = "CUSTOM"


# ---------------------------------------------------------------------------
# Standard Data Categories & Units
# ---------------------------------------------------------------------------


STANDARD_CATEGORIES: Dict[str, List[str]] = {
    "SCOPE_1": [
        "stationary_combustion",
        "mobile_combustion",
        "process_emissions",
        "fugitive_emissions",
        "refrigerant_leakage",
    ],
    "SCOPE_2": [
        "purchased_electricity",
        "purchased_steam",
        "purchased_heating",
        "purchased_cooling",
    ],
    "SCOPE_3": [
        "business_travel",
        "employee_commuting",
        "waste_generated",
        "purchased_goods_services",
        "upstream_transportation",
        "downstream_transportation",
    ],
    "ACTIVITY_DATA": [
        "natural_gas_consumption",
        "diesel_consumption",
        "petrol_consumption",
        "lpg_consumption",
        "electricity_consumption",
        "water_consumption",
        "waste_to_landfill",
        "waste_recycled",
        "business_travel_air",
        "business_travel_rail",
        "business_travel_road",
        "fleet_fuel",
        "refrigerant_type_quantity",
    ],
}

VALID_UNITS: Dict[str, List[str]] = {
    "energy": ["kWh", "MWh", "GJ", "therms", "MMBTU"],
    "fuel_volume": ["litres", "gallons", "m3"],
    "fuel_mass": ["kg", "tonnes", "lbs"],
    "distance": ["km", "miles"],
    "area": ["m2", "ft2"],
    "emissions": ["tCO2e", "kgCO2e"],
    "mass": ["kg", "tonnes"],
    "volume": ["m3", "litres"],
    "currency": ["USD", "EUR", "GBP"],
}

# Default range checks per category (min, max)
DEFAULT_RANGE_CHECKS: Dict[str, Tuple[Decimal, Decimal]] = {
    "natural_gas_consumption": (Decimal("0"), Decimal("100000000")),
    "diesel_consumption": (Decimal("0"), Decimal("50000000")),
    "petrol_consumption": (Decimal("0"), Decimal("50000000")),
    "electricity_consumption": (Decimal("0"), Decimal("500000000")),
    "water_consumption": (Decimal("0"), Decimal("100000000")),
    "waste_to_landfill": (Decimal("0"), Decimal("1000000")),
    "waste_recycled": (Decimal("0"), Decimal("1000000")),
    "business_travel_air": (Decimal("0"), Decimal("50000000")),
    "fleet_fuel": (Decimal("0"), Decimal("10000000")),
}

# Default YoY variance threshold (percentage)
DEFAULT_YOY_THRESHOLD: Decimal = Decimal("30")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class DataEntry(BaseModel):
    """A single data entry within a site submission.

    Represents one activity data point (e.g., electricity consumption
    for a given period) with quality metadata.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    entry_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this data entry.",
    )
    category: str = Field(
        ...,
        description="Data category (e.g., 'electricity_consumption').",
    )
    subcategory: str = Field(
        default="",
        description="Optional subcategory for finer classification.",
    )
    value: Decimal = Field(
        ...,
        description="Numeric value of the data point.",
    )
    unit: str = Field(
        ...,
        description="Unit of measurement (e.g., 'kWh', 'litres').",
    )
    source: str = Field(
        default="MANUAL",
        description="Data source type.",
    )
    methodology: Optional[str] = Field(
        None,
        description="Calculation or measurement methodology used.",
    )
    is_estimated: bool = Field(
        default=False,
        description="Whether this value is an estimate.",
    )
    estimation_method: Optional[str] = Field(
        None,
        description="Estimation method if is_estimated is True.",
    )
    quality_score: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Data quality score (1=lowest, 5=highest).",
    )
    notes: Optional[str] = Field(
        None,
        description="Additional notes or context.",
    )

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_value(cls, v: Any) -> Any:
        return Decimal(str(v))

    @field_validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        valid = {s.value for s in DataSource}
        upper = v.upper()
        if upper not in valid:
            logger.warning("Non-standard data source '%s'; accepted.", v)
        return upper


class ValidationRule(BaseModel):
    """A validation rule to apply to data entries."""
    model_config = ConfigDict(validate_default=True)

    rule_name: str = Field(
        ...,
        description="Unique name for the rule.",
    )
    rule_type: str = Field(
        ...,
        description="Type of validation rule.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rule parameters (e.g., min, max, threshold).",
    )
    severity: str = Field(
        default="WARNING",
        description="Severity if rule fails (ERROR, WARNING, INFO).",
    )
    applies_to_categories: List[str] = Field(
        default_factory=list,
        description="Categories this rule applies to (empty = all).",
    )

    @field_validator("rule_type")
    @classmethod
    def _validate_rule_type(cls, v: str) -> str:
        valid = {rt.value for rt in RuleType}
        if v.upper() not in valid:
            logger.warning("Non-standard rule type '%s'.", v)
        return v.upper()


class ValidationResult(BaseModel):
    """Result of a single validation check."""
    model_config = ConfigDict(validate_default=True)

    rule_name: str = Field(
        ..., description="The rule that was checked."
    )
    passed: bool = Field(
        ..., description="Whether the check passed."
    )
    message: str = Field(
        ..., description="Human-readable description of the result."
    )
    severity: str = Field(
        default="WARNING",
        description="Severity level of the finding.",
    )
    entry_id: Optional[str] = Field(
        None,
        description="The data entry this finding applies to.",
    )
    category: Optional[str] = Field(
        None,
        description="The data category this finding applies to.",
    )


class CollectionTemplate(BaseModel):
    """A data collection template for a specific site.

    Templates define what data categories and fields are expected
    from a site during a collection round.
    """
    model_config = ConfigDict(validate_default=True)

    template_id: str = Field(
        default_factory=_new_uuid,
        description="Unique template identifier.",
    )
    template_name: str = Field(
        ...,
        description="Human-readable template name.",
    )
    site_id: str = Field(
        ...,
        description="The site this template is generated for.",
    )
    data_categories: List[str] = Field(
        default_factory=list,
        description="Activity data categories to collect.",
    )
    required_fields: List[str] = Field(
        default_factory=list,
        description="Fields that must be provided.",
    )
    optional_fields: List[str] = Field(
        default_factory=list,
        description="Fields that are optional.",
    )
    validation_rules: List[ValidationRule] = Field(
        default_factory=list,
        description="Validation rules for this template.",
    )
    instructions: Optional[str] = Field(
        None,
        description="Instructions for the data collector.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the template was generated.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )


class SiteSubmission(BaseModel):
    """A data submission from a specific site for a collection round.

    Tracks the full lifecycle from DRAFT through APPROVED/REJECTED.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    submission_id: str = Field(
        default_factory=_new_uuid,
        description="Unique submission identifier.",
    )
    site_id: str = Field(
        ...,
        description="The site submitting data.",
    )
    round_id: str = Field(
        ...,
        description="The collection round this submission belongs to.",
    )
    period_start: date = Field(
        ...,
        description="Start of the reporting period.",
    )
    period_end: date = Field(
        ...,
        description="End of the reporting period.",
    )
    status: str = Field(
        default="DRAFT",
        description="Current submission status.",
    )
    data_entries: List[DataEntry] = Field(
        default_factory=list,
        description="Activity data entries.",
    )
    submitted_by: Optional[str] = Field(
        None,
        description="User who submitted the data.",
    )
    submitted_at: Optional[datetime] = Field(
        None,
        description="When the submission was made.",
    )
    reviewed_by: Optional[str] = Field(
        None,
        description="User who reviewed the submission.",
    )
    reviewed_at: Optional[datetime] = Field(
        None,
        description="When the review was completed.",
    )
    rejection_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons for rejection, if applicable.",
    )
    validation_errors: List[ValidationResult] = Field(
        default_factory=list,
        description="Validation findings.",
    )
    quality_flags: Dict[str, Any] = Field(
        default_factory=dict,
        description="Quality flags raised during validation.",
    )
    revision_number: int = Field(
        default=1,
        ge=1,
        description="Revision number of the submission.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the submission was created.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        valid = {s.value for s in SubmissionStatus}
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid submission status '{v}'. Must be one of {sorted(valid)}."
            )
        return v.upper()


class CollectionRound(BaseModel):
    """A time-bounded data collection round.

    Represents a coordinated data collection effort across multiple
    sites for a specific reporting period.
    """
    model_config = ConfigDict(validate_default=True)

    round_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for the round.",
    )
    period_type: str = Field(
        ...,
        description="Reporting period type (MONTHLY, QUARTERLY, etc.).",
    )
    period_start: date = Field(
        ...,
        description="Start of the reporting period.",
    )
    period_end: date = Field(
        ...,
        description="End of the reporting period.",
    )
    deadline: date = Field(
        ...,
        description="Submission deadline.",
    )
    site_ids: List[str] = Field(
        default_factory=list,
        description="Sites expected to submit data.",
    )
    sites_expected: int = Field(
        default=0,
        description="Total sites expected to submit.",
    )
    sites_submitted: int = Field(
        default=0,
        description="Sites that have submitted data.",
    )
    sites_approved: int = Field(
        default=0,
        description="Sites whose submissions are approved.",
    )
    status: str = Field(
        default="OPEN",
        description="Current round status.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the round was created.",
    )
    notes: Optional[str] = Field(
        None,
        description="Additional notes about the round.",
    )

    @field_validator("period_type")
    @classmethod
    def _validate_period_type(cls, v: str) -> str:
        valid = {pt.value for pt in PeriodType}
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid period_type '{v}'. Must be one of {sorted(valid)}."
            )
        return v.upper()

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        valid = {rs.value for rs in RoundStatus}
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid round status '{v}'. Must be one of {sorted(valid)}."
            )
        return v.upper()


class CollectionResult(BaseModel):
    """Summary result for a collection round.

    Aggregates collection progress, completeness, quality, and
    estimation metrics across all site submissions.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier.",
    )
    round: CollectionRound = Field(
        ...,
        description="The collection round.",
    )
    submissions: List[SiteSubmission] = Field(
        default_factory=list,
        description="All submissions for the round.",
    )
    completeness_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of expected sites that submitted.",
    )
    approval_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of submissions that are approved.",
    )
    estimation_pct: Decimal = Field(
        default=Decimal("0"),
        description="Percentage of data entries that are estimated.",
    )
    quality_avg: Decimal = Field(
        default=Decimal("0"),
        description="Average quality score across all entries.",
    )
    total_entries: int = Field(
        default=0,
        description="Total data entries across all submissions.",
    )
    total_estimated_entries: int = Field(
        default=0,
        description="Number of estimated entries.",
    )
    total_validation_errors: int = Field(
        default=0,
        description="Total ERROR-level validation findings.",
    )
    total_validation_warnings: int = Field(
        default=0,
        description="Total WARNING-level validation findings.",
    )
    sites_not_submitted: List[str] = Field(
        default_factory=list,
        description="Site IDs that have not yet submitted.",
    )
    categories_covered: List[str] = Field(
        default_factory=list,
        description="Distinct data categories with entries.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When this result was computed.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SiteDataCollectionEngine:
    """Manages GHG activity data collection across multiple sites.

    Provides collection round management, template generation,
    submission tracking, validation, approval workflows, and
    missing-data estimation. All numeric operations use Decimal
    arithmetic with SHA-256 provenance hashing.

    Attributes:
        _rounds: Internal dict mapping round_id to CollectionRound.
        _submissions: Internal dict mapping submission_id to SiteSubmission.
        _templates: Internal dict mapping template_id to CollectionTemplate.

    Example:
        >>> engine = SiteDataCollectionEngine()
        >>> round_ = engine.create_collection_round(
        ...     period_type="QUARTERLY",
        ...     start=date(2025, 1, 1),
        ...     end=date(2025, 3, 31),
        ...     deadline=date(2025, 4, 30),
        ...     site_ids=["site-001", "site-002"],
        ... )
        >>> submission = engine.submit_site_data(
        ...     site_id="site-001",
        ...     round_id=round_.round_id,
        ...     data_entries=[...],
        ... )
    """

    def __init__(self) -> None:
        """Initialise the SiteDataCollectionEngine with empty state."""
        self._rounds: Dict[str, CollectionRound] = {}
        self._submissions: Dict[str, SiteSubmission] = {}
        self._templates: Dict[str, CollectionTemplate] = {}
        self._custom_rules: List[ValidationRule] = []
        logger.info("SiteDataCollectionEngine v%s initialised.", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Collection Rounds
    # ------------------------------------------------------------------

    def create_collection_round(
        self,
        period_type: str,
        start: date,
        end: date,
        deadline: date,
        site_ids: List[str],
        notes: Optional[str] = None,
    ) -> CollectionRound:
        """Create a new data collection round.

        Args:
            period_type: MONTHLY, QUARTERLY, SEMI_ANNUAL, or ANNUAL.
            start: Start date of the reporting period.
            end: End date of the reporting period.
            deadline: Submission deadline.
            site_ids: List of site IDs expected to submit data.
            notes: Optional notes about the round.

        Returns:
            The created CollectionRound.

        Raises:
            ValueError: If dates are inconsistent.
        """
        if end <= start:
            raise ValueError(
                f"period_end ({end}) must be after period_start ({start})."
            )
        if deadline < end:
            raise ValueError(
                f"deadline ({deadline}) must be on or after period_end ({end})."
            )
        if not site_ids:
            raise ValueError("At least one site_id is required.")

        logger.info(
            "Creating %s collection round: %s to %s, deadline %s, %d sites.",
            period_type, start, end, deadline, len(site_ids),
        )

        round_ = CollectionRound(
            period_type=period_type,
            period_start=start,
            period_end=end,
            deadline=deadline,
            site_ids=list(site_ids),
            sites_expected=len(site_ids),
            status=RoundStatus.OPEN.value,
            notes=notes,
        )
        self._rounds[round_.round_id] = round_

        logger.info("Collection round '%s' created.", round_.round_id)
        return round_

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def generate_template(
        self,
        site_id: str,
        categories: Optional[List[str]] = None,
        include_scope3: bool = False,
    ) -> CollectionTemplate:
        """Generate a data collection template for a specific site.

        Creates a template with the required data categories,
        fields, and validation rules for the site.

        Args:
            site_id: The site to generate a template for.
            categories: Custom list of categories. If None, uses
                standard SCOPE_1 + SCOPE_2 + ACTIVITY_DATA categories.
            include_scope3: If True and categories is None, also
                include SCOPE_3 categories.

        Returns:
            A CollectionTemplate with all fields and rules.
        """
        logger.info("Generating collection template for site '%s'.", site_id)

        if categories is None:
            categories = (
                STANDARD_CATEGORIES["SCOPE_1"]
                + STANDARD_CATEGORIES["SCOPE_2"]
                + STANDARD_CATEGORIES["ACTIVITY_DATA"]
            )
            if include_scope3:
                categories += STANDARD_CATEGORIES["SCOPE_3"]

        # Required fields per entry
        required_fields = ["category", "value", "unit", "source"]
        optional_fields = [
            "subcategory",
            "methodology",
            "is_estimated",
            "estimation_method",
            "quality_score",
            "notes",
        ]

        # Build validation rules
        rules: List[ValidationRule] = []

        # Range checks for known categories
        for cat in categories:
            if cat in DEFAULT_RANGE_CHECKS:
                min_val, max_val = DEFAULT_RANGE_CHECKS[cat]
                rules.append(ValidationRule(
                    rule_name=f"range_{cat}",
                    rule_type=RuleType.RANGE_CHECK.value,
                    params={"min": str(min_val), "max": str(max_val)},
                    severity=ValidationSeverity.WARNING.value,
                    applies_to_categories=[cat],
                ))

        # YoY variance check
        rules.append(ValidationRule(
            rule_name="yoy_variance_all",
            rule_type=RuleType.YOY_VARIANCE.value,
            params={"threshold_pct": str(DEFAULT_YOY_THRESHOLD)},
            severity=ValidationSeverity.WARNING.value,
        ))

        # Completeness check
        rules.append(ValidationRule(
            rule_name="completeness_check",
            rule_type=RuleType.COMPLETENESS.value,
            params={"required_categories": categories},
            severity=ValidationSeverity.ERROR.value,
        ))

        # Add custom rules
        for cr in self._custom_rules:
            rules.append(cr)

        template_name = f"Collection Template - {site_id}"

        template = CollectionTemplate(
            template_name=template_name,
            site_id=site_id,
            data_categories=categories,
            required_fields=required_fields,
            optional_fields=optional_fields,
            validation_rules=rules,
            instructions=(
                f"Please provide activity data for the following categories: "
                f"{', '.join(categories[:5])}{'...' if len(categories) > 5 else ''}. "
                f"All values should be actual measured or invoiced data where possible. "
                f"Estimated values must be flagged with is_estimated=True."
            ),
        )
        template.provenance_hash = _compute_hash(template)
        self._templates[template.template_id] = template

        logger.info(
            "Template '%s' generated with %d categories and %d rules.",
            template.template_id,
            len(categories),
            len(rules),
        )
        return template

    # ------------------------------------------------------------------
    # Submissions
    # ------------------------------------------------------------------

    def submit_site_data(
        self,
        site_id: str,
        round_id: str,
        data_entries: List[Union[Dict[str, Any], DataEntry]],
        submitted_by: Optional[str] = None,
    ) -> SiteSubmission:
        """Submit activity data for a site within a collection round.

        Creates a submission, validates the data, and stores it. The
        submission starts in SUBMITTED status after passing basic
        validation, or DRAFT if errors are found.

        Args:
            site_id: The site submitting data.
            round_id: The collection round.
            data_entries: List of data entries (dicts or DataEntry objects).
            submitted_by: User identifier of the submitter.

        Returns:
            The created SiteSubmission with validation results.

        Raises:
            KeyError: If round_id is not found.
            ValueError: If the round is not open for submissions.
        """
        if round_id not in self._rounds:
            raise KeyError(f"Collection round '{round_id}' not found.")

        round_ = self._rounds[round_id]
        if round_.status not in (
            RoundStatus.OPEN.value,
            RoundStatus.EXTENDED.value,
        ):
            raise ValueError(
                f"Round '{round_id}' is not open (status={round_.status})."
            )

        logger.info(
            "Processing submission from site '%s' for round '%s'.",
            site_id, round_id,
        )

        # Parse data entries
        parsed_entries: List[DataEntry] = []
        for entry in data_entries:
            if isinstance(entry, dict):
                parsed_entries.append(DataEntry(**entry))
            else:
                parsed_entries.append(entry)

        # Check for existing submission and increment revision
        revision = 1
        for sub in self._submissions.values():
            if sub.site_id == site_id and sub.round_id == round_id:
                revision = max(revision, sub.revision_number + 1)

        now = _utcnow()
        submission = SiteSubmission(
            site_id=site_id,
            round_id=round_id,
            period_start=round_.period_start,
            period_end=round_.period_end,
            data_entries=parsed_entries,
            submitted_by=submitted_by,
            submitted_at=now,
            revision_number=revision,
        )

        # Validate
        validation_results = self.validate_submission(submission)
        submission.validation_errors = validation_results

        # Count errors
        error_count = sum(
            1 for vr in validation_results
            if not vr.passed and vr.severity == ValidationSeverity.ERROR.value
        )
        warning_count = sum(
            1 for vr in validation_results
            if not vr.passed and vr.severity == ValidationSeverity.WARNING.value
        )

        submission.quality_flags = {
            "error_count": error_count,
            "warning_count": warning_count,
            "total_entries": len(parsed_entries),
            "estimated_entries": sum(1 for e in parsed_entries if e.is_estimated),
        }

        if error_count > 0:
            submission.status = SubmissionStatus.DRAFT.value
            logger.warning(
                "Submission has %d error(s); status set to DRAFT.", error_count
            )
        else:
            submission.status = SubmissionStatus.SUBMITTED.value

        submission.provenance_hash = _compute_hash(submission)
        self._submissions[submission.submission_id] = submission

        # Update round counters
        self._update_round_counters(round_id)

        logger.info(
            "Submission '%s' created (rev=%d, status=%s, entries=%d, errors=%d, warnings=%d).",
            submission.submission_id,
            revision,
            submission.status,
            len(parsed_entries),
            error_count,
            warning_count,
        )
        return submission

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_submission(
        self,
        submission: SiteSubmission,
        additional_rules: Optional[List[ValidationRule]] = None,
        historical_data: Optional[Dict[str, Decimal]] = None,
    ) -> List[ValidationResult]:
        """Validate a site submission against all applicable rules.

        Runs range checks, year-on-year variance, unit validation,
        and completeness checks. Returns a list of validation results.

        Args:
            submission: The submission to validate.
            additional_rules: Extra rules to apply.
            historical_data: Previous period values by category for
                YoY comparison. Keys are categories, values are
                previous period totals.

        Returns:
            List of ValidationResult objects.
        """
        logger.info(
            "Validating submission '%s' (%d entries).",
            submission.submission_id,
            len(submission.data_entries),
        )

        results: List[ValidationResult] = []
        historical_data = historical_data or {}

        # 1. Range checks
        for entry in submission.data_entries:
            range_results = self._validate_range(entry)
            results.extend(range_results)

        # 2. Unit checks
        for entry in submission.data_entries:
            unit_results = self._validate_unit(entry)
            results.extend(unit_results)

        # 3. Negative value check
        for entry in submission.data_entries:
            if entry.value < Decimal("0"):
                results.append(ValidationResult(
                    rule_name="non_negative_check",
                    passed=False,
                    message=(
                        f"Entry '{entry.entry_id}' has negative value "
                        f"({entry.value}) for category '{entry.category}'."
                    ),
                    severity=ValidationSeverity.ERROR.value,
                    entry_id=entry.entry_id,
                    category=entry.category,
                ))

        # 4. YoY variance
        if historical_data:
            # Aggregate current by category
            current_by_cat: Dict[str, Decimal] = {}
            for entry in submission.data_entries:
                current_by_cat[entry.category] = (
                    current_by_cat.get(entry.category, Decimal("0")) + entry.value
                )
            for cat, current_val in current_by_cat.items():
                if cat in historical_data:
                    yoy_result = self.calculate_yoy_variance(
                        current_val, historical_data[cat]
                    )
                    variance_pct = yoy_result["variance_pct"]
                    if abs(variance_pct) > DEFAULT_YOY_THRESHOLD:
                        results.append(ValidationResult(
                            rule_name="yoy_variance",
                            passed=False,
                            message=(
                                f"Category '{cat}' has {variance_pct}% YoY "
                                f"variance (threshold: +/-{DEFAULT_YOY_THRESHOLD}%)."
                            ),
                            severity=ValidationSeverity.WARNING.value,
                            category=cat,
                        ))
                    else:
                        results.append(ValidationResult(
                            rule_name="yoy_variance",
                            passed=True,
                            message=(
                                f"Category '{cat}' YoY variance "
                                f"{variance_pct}% within threshold."
                            ),
                            severity=ValidationSeverity.INFO.value,
                            category=cat,
                        ))

        # 5. Completeness check
        completeness_results = self._validate_completeness(
            submission.data_entries
        )
        results.extend(completeness_results)

        # 6. Estimation flagging
        estimated_entries = [e for e in submission.data_entries if e.is_estimated]
        if estimated_entries:
            for entry in estimated_entries:
                if not entry.estimation_method:
                    results.append(ValidationResult(
                        rule_name="estimation_method_required",
                        passed=False,
                        message=(
                            f"Entry '{entry.entry_id}' is estimated but "
                            f"no estimation_method specified."
                        ),
                        severity=ValidationSeverity.WARNING.value,
                        entry_id=entry.entry_id,
                        category=entry.category,
                    ))

        # 7. Apply additional custom rules
        if additional_rules:
            for rule in additional_rules:
                rule_results = self._apply_custom_rule(
                    rule, submission.data_entries
                )
                results.extend(rule_results)

        passed_count = sum(1 for r in results if r.passed)
        failed_count = sum(1 for r in results if not r.passed)
        logger.info(
            "Validation complete: %d passed, %d failed.",
            passed_count, failed_count,
        )
        return results

    def _validate_range(self, entry: DataEntry) -> List[ValidationResult]:
        """Check if entry value falls within expected range.

        Args:
            entry: The data entry to check.

        Returns:
            List with one result if a range check exists for the category.
        """
        results: List[ValidationResult] = []
        if entry.category in DEFAULT_RANGE_CHECKS:
            min_val, max_val = DEFAULT_RANGE_CHECKS[entry.category]
            if entry.value < min_val or entry.value > max_val:
                results.append(ValidationResult(
                    rule_name=f"range_{entry.category}",
                    passed=False,
                    message=(
                        f"Value {entry.value} for '{entry.category}' is "
                        f"outside expected range [{min_val}, {max_val}]."
                    ),
                    severity=ValidationSeverity.WARNING.value,
                    entry_id=entry.entry_id,
                    category=entry.category,
                ))
            else:
                results.append(ValidationResult(
                    rule_name=f"range_{entry.category}",
                    passed=True,
                    message=(
                        f"Value {entry.value} for '{entry.category}' "
                        f"within expected range."
                    ),
                    severity=ValidationSeverity.INFO.value,
                    entry_id=entry.entry_id,
                    category=entry.category,
                ))
        return results

    def _validate_unit(self, entry: DataEntry) -> List[ValidationResult]:
        """Validate that the unit is recognised.

        Args:
            entry: The data entry to check.

        Returns:
            List with one result.
        """
        all_units: set = set()
        for units in VALID_UNITS.values():
            all_units.update(units)

        if entry.unit not in all_units:
            return [ValidationResult(
                rule_name="unit_check",
                passed=False,
                message=(
                    f"Unit '{entry.unit}' for entry '{entry.entry_id}' "
                    f"is not a recognised standard unit."
                ),
                severity=ValidationSeverity.WARNING.value,
                entry_id=entry.entry_id,
                category=entry.category,
            )]
        return [ValidationResult(
            rule_name="unit_check",
            passed=True,
            message=f"Unit '{entry.unit}' is valid.",
            severity=ValidationSeverity.INFO.value,
            entry_id=entry.entry_id,
            category=entry.category,
        )]

    def _validate_completeness(
        self, entries: List[DataEntry],
    ) -> List[ValidationResult]:
        """Check that key activity data categories are represented.

        Args:
            entries: List of data entries.

        Returns:
            Validation results for completeness checks.
        """
        results: List[ValidationResult] = []
        present_categories = {e.category for e in entries}

        # Core categories that should almost always be present
        core_categories = [
            "electricity_consumption",
            "natural_gas_consumption",
        ]

        for cat in core_categories:
            if cat in present_categories:
                results.append(ValidationResult(
                    rule_name=f"completeness_{cat}",
                    passed=True,
                    message=f"Core category '{cat}' is present.",
                    severity=ValidationSeverity.INFO.value,
                    category=cat,
                ))
            else:
                results.append(ValidationResult(
                    rule_name=f"completeness_{cat}",
                    passed=False,
                    message=(
                        f"Core category '{cat}' is missing. This is "
                        f"typically required for GHG reporting."
                    ),
                    severity=ValidationSeverity.WARNING.value,
                    category=cat,
                ))

        return results

    def _apply_custom_rule(
        self,
        rule: ValidationRule,
        entries: List[DataEntry],
    ) -> List[ValidationResult]:
        """Apply a custom validation rule to data entries.

        Args:
            rule: The custom rule.
            entries: Data entries to validate.

        Returns:
            Validation results.
        """
        results: List[ValidationResult] = []

        # Filter entries by applicable categories
        if rule.applies_to_categories:
            applicable = [
                e for e in entries
                if e.category in rule.applies_to_categories
            ]
        else:
            applicable = entries

        if rule.rule_type == RuleType.RANGE_CHECK.value:
            min_val = _decimal(rule.params.get("min", "0"))
            max_val = _decimal(rule.params.get("max", "999999999"))
            for entry in applicable:
                if entry.value < min_val or entry.value > max_val:
                    results.append(ValidationResult(
                        rule_name=rule.rule_name,
                        passed=False,
                        message=(
                            f"Custom range: value {entry.value} outside "
                            f"[{min_val}, {max_val}] for '{entry.category}'."
                        ),
                        severity=rule.severity,
                        entry_id=entry.entry_id,
                        category=entry.category,
                    ))

        elif rule.rule_type == RuleType.CROSS_FIELD.value:
            # Cross-field rules compare two categories
            cat_a = rule.params.get("category_a", "")
            cat_b = rule.params.get("category_b", "")
            max_ratio = _decimal(rule.params.get("max_ratio", "100"))
            sum_a = sum(
                (e.value for e in entries if e.category == cat_a),
                Decimal("0"),
            )
            sum_b = sum(
                (e.value for e in entries if e.category == cat_b),
                Decimal("0"),
            )
            if sum_b > Decimal("0"):
                ratio = _safe_divide(sum_a, sum_b)
                if ratio > max_ratio:
                    results.append(ValidationResult(
                        rule_name=rule.rule_name,
                        passed=False,
                        message=(
                            f"Cross-field: ratio of '{cat_a}' to '{cat_b}' "
                            f"is {_round2(ratio)}, exceeds max {max_ratio}."
                        ),
                        severity=rule.severity,
                    ))

        return results

    # ------------------------------------------------------------------
    # Approval Workflow
    # ------------------------------------------------------------------

    def approve_submission(
        self,
        submission: SiteSubmission,
        reviewer: str,
    ) -> SiteSubmission:
        """Approve a site data submission.

        Transitions the submission from SUBMITTED or UNDER_REVIEW
        to APPROVED. Records the reviewer and timestamp.

        Args:
            submission: The submission to approve.
            reviewer: User ID of the reviewer.

        Returns:
            The updated SiteSubmission.

        Raises:
            ValueError: If submission is not in an approvable state.
        """
        approvable = {
            SubmissionStatus.SUBMITTED.value,
            SubmissionStatus.UNDER_REVIEW.value,
        }
        if submission.status not in approvable:
            raise ValueError(
                f"Cannot approve submission in '{submission.status}' status. "
                f"Must be one of {sorted(approvable)}."
            )

        logger.info(
            "Approving submission '%s' by reviewer '%s'.",
            submission.submission_id,
            reviewer,
        )

        now = _utcnow()
        updated = submission.model_copy(update={
            "status": SubmissionStatus.APPROVED.value,
            "reviewed_by": reviewer,
            "reviewed_at": now,
        })
        updated.provenance_hash = _compute_hash(updated)
        self._submissions[updated.submission_id] = updated

        # Update round counters
        self._update_round_counters(updated.round_id)

        logger.info("Submission '%s' approved.", updated.submission_id)
        return updated

    def reject_submission(
        self,
        submission: SiteSubmission,
        reviewer: str,
        reasons: List[str],
    ) -> SiteSubmission:
        """Reject a site data submission with reasons.

        Transitions the submission to REJECTED status and records
        the rejection reasons for the submitter to address.

        Args:
            submission: The submission to reject.
            reviewer: User ID of the reviewer.
            reasons: List of rejection reasons.

        Returns:
            The updated SiteSubmission.

        Raises:
            ValueError: If submission is not in a rejectable state.
            ValueError: If no reasons are provided.
        """
        rejectable = {
            SubmissionStatus.SUBMITTED.value,
            SubmissionStatus.UNDER_REVIEW.value,
        }
        if submission.status not in rejectable:
            raise ValueError(
                f"Cannot reject submission in '{submission.status}' status."
            )
        if not reasons:
            raise ValueError("At least one rejection reason is required.")

        logger.info(
            "Rejecting submission '%s' by reviewer '%s' with %d reason(s).",
            submission.submission_id,
            reviewer,
            len(reasons),
        )

        now = _utcnow()
        updated = submission.model_copy(update={
            "status": SubmissionStatus.REJECTED.value,
            "reviewed_by": reviewer,
            "reviewed_at": now,
            "rejection_reasons": list(reasons),
        })
        updated.provenance_hash = _compute_hash(updated)
        self._submissions[updated.submission_id] = updated

        logger.info("Submission '%s' rejected.", updated.submission_id)
        return updated

    # ------------------------------------------------------------------
    # Collection Status
    # ------------------------------------------------------------------

    def get_collection_status(
        self,
        round_: CollectionRound,
        submissions: Optional[List[SiteSubmission]] = None,
    ) -> CollectionResult:
        """Calculate collection status for a round.

        Aggregates submission counts, completeness, quality scores,
        and estimation percentages across all submissions.

        Args:
            round_: The collection round.
            submissions: Submissions for the round. If None, fetches
                from internal storage.

        Returns:
            CollectionResult with comprehensive metrics.
        """
        if submissions is None:
            submissions = [
                s for s in self._submissions.values()
                if s.round_id == round_.round_id
            ]

        logger.info(
            "Computing collection status for round '%s' (%d submissions).",
            round_.round_id,
            len(submissions),
        )

        # Aggregate metrics
        total_entries = 0
        total_estimated = 0
        total_quality_sum = Decimal("0")
        total_errors = 0
        total_warnings = 0
        all_categories: set = set()
        submitted_site_ids: set = set()

        for sub in submissions:
            submitted_site_ids.add(sub.site_id)
            for entry in sub.data_entries:
                total_entries += 1
                total_quality_sum += _decimal(entry.quality_score)
                all_categories.add(entry.category)
                if entry.is_estimated:
                    total_estimated += 1

            for vr in sub.validation_errors:
                if not vr.passed:
                    if vr.severity == ValidationSeverity.ERROR.value:
                        total_errors += 1
                    elif vr.severity == ValidationSeverity.WARNING.value:
                        total_warnings += 1

        # Compute percentages
        expected = _decimal(round_.sites_expected) if round_.sites_expected > 0 else Decimal("1")
        completeness_pct = _round2(
            _safe_divide(_decimal(len(submitted_site_ids)), expected) * Decimal("100")
        )

        approved_count = sum(
            1 for s in submissions
            if s.status == SubmissionStatus.APPROVED.value
        )
        approval_pct = _round2(
            _safe_divide(
                _decimal(approved_count),
                _decimal(len(submissions)) if submissions else Decimal("1"),
            ) * Decimal("100")
        )

        estimation_pct = _round2(
            _safe_divide(
                _decimal(total_estimated),
                _decimal(total_entries) if total_entries > 0 else Decimal("1"),
            ) * Decimal("100")
        )

        quality_avg = _round2(
            _safe_divide(
                total_quality_sum,
                _decimal(total_entries) if total_entries > 0 else Decimal("1"),
            )
        )

        # Sites not submitted
        expected_set = set(round_.site_ids) if round_.site_ids else set()
        not_submitted = sorted(expected_set - submitted_site_ids)

        # Update round counters
        updated_round = round_.model_copy(update={
            "sites_submitted": len(submitted_site_ids),
            "sites_approved": approved_count,
        })

        result = CollectionResult(
            round=updated_round,
            submissions=submissions,
            completeness_pct=completeness_pct,
            approval_pct=approval_pct,
            estimation_pct=estimation_pct,
            quality_avg=quality_avg,
            total_entries=total_entries,
            total_estimated_entries=total_estimated,
            total_validation_errors=total_errors,
            total_validation_warnings=total_warnings,
            sites_not_submitted=not_submitted,
            categories_covered=sorted(all_categories),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Round '%s' status: completeness=%s%%, approval=%s%%, "
            "quality_avg=%s, entries=%d, estimated=%d.",
            round_.round_id,
            completeness_pct,
            approval_pct,
            quality_avg,
            total_entries,
            total_estimated,
        )
        return result

    # ------------------------------------------------------------------
    # Missing Data Estimation
    # ------------------------------------------------------------------

    def estimate_missing_data(
        self,
        site_id: str,
        category: str,
        method: str,
        historical: Optional[List[Dict[str, Any]]] = None,
        proxy_data: Optional[Dict[str, Any]] = None,
        unit: str = "kWh",
    ) -> List[DataEntry]:
        """Estimate missing data for a site and category.

        Supports multiple estimation methods with full auditability.

        Methods:
            PRIOR_YEAR: Uses the most recent historical value.
            EXTRAPOLATION: Extrapolates from partial-period data.
            SECTOR_AVERAGE: Uses a sector average proxy.
            LINEAR_INTERPOLATION: Interpolates between known points.
            INTENSITY_BASED: Uses an intensity metric (e.g., per m2).
            PROXY_DATA: Uses proxy data from a similar site.

        Args:
            site_id: The site missing data.
            category: The data category.
            method: The estimation method to use.
            historical: List of historical data dicts with
                keys 'period', 'value', 'unit'.
            proxy_data: Proxy data dict with keys 'value', 'unit',
                'source_site_id', 'scaling_factor'.
            unit: Default unit for estimated entries.

        Returns:
            List of estimated DataEntry objects.

        Raises:
            ValueError: If method is unsupported or insufficient data.
        """
        method_upper = method.upper()
        valid_methods = {em.value for em in EstimationMethod}
        if method_upper not in valid_methods:
            raise ValueError(
                f"Unsupported estimation method '{method}'. "
                f"Must be one of {sorted(valid_methods)}."
            )

        logger.info(
            "Estimating '%s' for site '%s' using method '%s'.",
            category, site_id, method_upper,
        )

        historical = historical or []
        entries: List[DataEntry] = []

        if method_upper == EstimationMethod.PRIOR_YEAR.value:
            entries = self._estimate_prior_year(
                site_id, category, historical, unit
            )

        elif method_upper == EstimationMethod.EXTRAPOLATION.value:
            entries = self._estimate_extrapolation(
                site_id, category, historical, unit
            )

        elif method_upper == EstimationMethod.SECTOR_AVERAGE.value:
            entries = self._estimate_sector_average(
                site_id, category, proxy_data, unit
            )

        elif method_upper == EstimationMethod.LINEAR_INTERPOLATION.value:
            entries = self._estimate_linear_interpolation(
                site_id, category, historical, unit
            )

        elif method_upper == EstimationMethod.INTENSITY_BASED.value:
            entries = self._estimate_intensity_based(
                site_id, category, proxy_data, unit
            )

        elif method_upper == EstimationMethod.PROXY_DATA.value:
            entries = self._estimate_proxy(
                site_id, category, proxy_data, unit
            )

        logger.info(
            "Estimated %d entry(ies) for site '%s', category '%s'.",
            len(entries), site_id, category,
        )
        return entries

    def _estimate_prior_year(
        self,
        site_id: str,
        category: str,
        historical: List[Dict[str, Any]],
        unit: str,
    ) -> List[DataEntry]:
        """Estimate using the most recent historical value.

        Args:
            site_id: Site identifier.
            category: Data category.
            historical: Historical data list.
            unit: Unit of measurement.

        Returns:
            Single DataEntry with the prior year value.
        """
        if not historical:
            raise ValueError(
                f"PRIOR_YEAR estimation requires historical data for "
                f"site '{site_id}', category '{category}'."
            )

        # Sort by period descending, take the most recent
        sorted_hist = sorted(
            historical, key=lambda x: str(x.get("period", "")), reverse=True
        )
        most_recent = sorted_hist[0]
        value = _decimal(most_recent.get("value", "0"))

        return [DataEntry(
            category=category,
            subcategory="",
            value=value,
            unit=most_recent.get("unit", unit),
            source=DataSource.ESTIMATED.value,
            methodology=f"PRIOR_YEAR from period {most_recent.get('period', 'unknown')}",
            is_estimated=True,
            estimation_method=EstimationMethod.PRIOR_YEAR.value,
            quality_score=2,
            notes=f"Estimated using prior year data. Original site: {site_id}.",
        )]

    def _estimate_extrapolation(
        self,
        site_id: str,
        category: str,
        historical: List[Dict[str, Any]],
        unit: str,
    ) -> List[DataEntry]:
        """Extrapolate from partial-period data to full period.

        If 9 months of data yields X, then 12 months = X * (12/9).

        Args:
            site_id: Site identifier.
            category: Data category.
            historical: Partial period data.
            unit: Unit of measurement.

        Returns:
            Single DataEntry with the extrapolated value.
        """
        if not historical:
            raise ValueError(
                f"EXTRAPOLATION requires partial period data for "
                f"site '{site_id}', category '{category}'."
            )

        # Sum all known partial values
        partial_sum = Decimal("0")
        months_covered = 0
        for record in historical:
            partial_sum += _decimal(record.get("value", "0"))
            months_covered += int(record.get("months", 1))

        if months_covered == 0:
            raise ValueError("Cannot extrapolate with zero months covered.")

        # Extrapolate to 12 months
        target_months = Decimal("12")
        extrapolated = _round2(
            partial_sum * _safe_divide(target_months, _decimal(months_covered))
        )

        return [DataEntry(
            category=category,
            subcategory="",
            value=extrapolated,
            unit=unit,
            source=DataSource.ESTIMATED.value,
            methodology=(
                f"EXTRAPOLATION: {partial_sum} over {months_covered} months "
                f"-> {extrapolated} over 12 months"
            ),
            is_estimated=True,
            estimation_method=EstimationMethod.EXTRAPOLATION.value,
            quality_score=2,
            notes=f"Extrapolated from {months_covered} months of data.",
        )]

    def _estimate_sector_average(
        self,
        site_id: str,
        category: str,
        proxy_data: Optional[Dict[str, Any]],
        unit: str,
    ) -> List[DataEntry]:
        """Estimate using a sector average value.

        Args:
            site_id: Site identifier.
            category: Data category.
            proxy_data: Must contain 'sector_average' and optionally
                'scaling_factor'.
            unit: Unit of measurement.

        Returns:
            Single DataEntry with the sector average value.
        """
        if not proxy_data or "sector_average" not in proxy_data:
            raise ValueError(
                f"SECTOR_AVERAGE requires proxy_data with 'sector_average' key."
            )

        sector_avg = _decimal(proxy_data["sector_average"])
        scaling = _decimal(proxy_data.get("scaling_factor", "1"))
        estimated_value = _round2(sector_avg * scaling)

        return [DataEntry(
            category=category,
            subcategory="",
            value=estimated_value,
            unit=proxy_data.get("unit", unit),
            source=DataSource.ESTIMATED.value,
            methodology=(
                f"SECTOR_AVERAGE: {sector_avg} * scaling {scaling} = {estimated_value}"
            ),
            is_estimated=True,
            estimation_method=EstimationMethod.SECTOR_AVERAGE.value,
            quality_score=1,
            notes=f"Based on sector average. Source: {proxy_data.get('source', 'unknown')}.",
        )]

    def _estimate_linear_interpolation(
        self,
        site_id: str,
        category: str,
        historical: List[Dict[str, Any]],
        unit: str,
    ) -> List[DataEntry]:
        """Linearly interpolate between two known data points.

        Args:
            site_id: Site identifier.
            category: Data category.
            historical: Must contain at least 2 records with
                'period_index' and 'value'.
            unit: Unit of measurement.

        Returns:
            DataEntries for interpolated gaps.
        """
        if len(historical) < 2:
            raise ValueError(
                f"LINEAR_INTERPOLATION requires at least 2 data points."
            )

        sorted_hist = sorted(
            historical, key=lambda x: int(x.get("period_index", 0))
        )

        entries: List[DataEntry] = []
        for i in range(len(sorted_hist) - 1):
            p1 = sorted_hist[i]
            p2 = sorted_hist[i + 1]
            idx1 = int(p1.get("period_index", 0))
            idx2 = int(p2.get("period_index", 0))
            val1 = _decimal(p1.get("value", "0"))
            val2 = _decimal(p2.get("value", "0"))

            # Interpolate for gaps
            for gap_idx in range(idx1 + 1, idx2):
                fraction = _safe_divide(
                    _decimal(gap_idx - idx1),
                    _decimal(idx2 - idx1),
                )
                interpolated = _round2(val1 + fraction * (val2 - val1))
                entries.append(DataEntry(
                    category=category,
                    subcategory="",
                    value=interpolated,
                    unit=unit,
                    source=DataSource.ESTIMATED.value,
                    methodology=(
                        f"LINEAR_INTERPOLATION between index {idx1} "
                        f"({val1}) and {idx2} ({val2}), at index {gap_idx}"
                    ),
                    is_estimated=True,
                    estimation_method=EstimationMethod.LINEAR_INTERPOLATION.value,
                    quality_score=2,
                    notes=f"Interpolated gap at period index {gap_idx}.",
                ))

        return entries

    def _estimate_intensity_based(
        self,
        site_id: str,
        category: str,
        proxy_data: Optional[Dict[str, Any]],
        unit: str,
    ) -> List[DataEntry]:
        """Estimate using an intensity metric (e.g., kWh per m2).

        Args:
            site_id: Site identifier.
            category: Data category.
            proxy_data: Must contain 'intensity_value',
                'intensity_unit', and 'activity_metric'.
            unit: Unit of measurement.

        Returns:
            Single DataEntry with intensity-based estimate.
        """
        if not proxy_data:
            raise ValueError("INTENSITY_BASED requires proxy_data.")

        intensity = _decimal(proxy_data.get("intensity_value", "0"))
        activity = _decimal(proxy_data.get("activity_metric", "0"))
        estimated_value = _round2(intensity * activity)

        return [DataEntry(
            category=category,
            subcategory="",
            value=estimated_value,
            unit=proxy_data.get("unit", unit),
            source=DataSource.ESTIMATED.value,
            methodology=(
                f"INTENSITY_BASED: {intensity} "
                f"{proxy_data.get('intensity_unit', 'per unit')} "
                f"* {activity} = {estimated_value}"
            ),
            is_estimated=True,
            estimation_method=EstimationMethod.INTENSITY_BASED.value,
            quality_score=2,
            notes=f"Intensity-based estimate for site {site_id}.",
        )]

    def _estimate_proxy(
        self,
        site_id: str,
        category: str,
        proxy_data: Optional[Dict[str, Any]],
        unit: str,
    ) -> List[DataEntry]:
        """Estimate using proxy data from a similar site.

        Args:
            site_id: Site identifier.
            category: Data category.
            proxy_data: Must contain 'value', 'source_site_id', and
                optionally 'scaling_factor'.
            unit: Unit of measurement.

        Returns:
            Single DataEntry with proxy-based estimate.
        """
        if not proxy_data or "value" not in proxy_data:
            raise ValueError(
                "PROXY_DATA requires proxy_data with 'value' key."
            )

        proxy_value = _decimal(proxy_data["value"])
        scaling = _decimal(proxy_data.get("scaling_factor", "1"))
        source_site = proxy_data.get("source_site_id", "unknown")
        estimated_value = _round2(proxy_value * scaling)

        return [DataEntry(
            category=category,
            subcategory="",
            value=estimated_value,
            unit=proxy_data.get("unit", unit),
            source=DataSource.ESTIMATED.value,
            methodology=(
                f"PROXY_DATA from site '{source_site}': "
                f"{proxy_value} * scaling {scaling} = {estimated_value}"
            ),
            is_estimated=True,
            estimation_method=EstimationMethod.PROXY_DATA.value,
            quality_score=1,
            notes=f"Proxy from site '{source_site}', scaling={scaling}.",
        )]

    # ------------------------------------------------------------------
    # YoY Variance
    # ------------------------------------------------------------------

    def calculate_yoy_variance(
        self,
        current_value: Union[Decimal, str, int, float],
        previous_value: Union[Decimal, str, int, float],
    ) -> Dict[str, Any]:
        """Calculate year-on-year variance between two values.

        Args:
            current_value: Current period value.
            previous_value: Previous period value.

        Returns:
            Dictionary with keys: current, previous, absolute_change,
            variance_pct, direction.
        """
        current = _decimal(current_value)
        previous = _decimal(previous_value)
        absolute_change = current - previous

        if previous == Decimal("0"):
            variance_pct = Decimal("100") if current > Decimal("0") else Decimal("0")
        else:
            variance_pct = _round2(
                _safe_divide(absolute_change, previous) * Decimal("100")
            )

        if absolute_change > Decimal("0"):
            direction = "INCREASE"
        elif absolute_change < Decimal("0"):
            direction = "DECREASE"
        else:
            direction = "UNCHANGED"

        return {
            "current": current,
            "previous": previous,
            "absolute_change": absolute_change,
            "variance_pct": variance_pct,
            "direction": direction,
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _update_round_counters(self, round_id: str) -> None:
        """Recalculate round submission and approval counters.

        Args:
            round_id: The round to update.
        """
        if round_id not in self._rounds:
            return

        round_ = self._rounds[round_id]
        submissions = [
            s for s in self._submissions.values()
            if s.round_id == round_id
        ]

        submitted_sites = {s.site_id for s in submissions}
        approved_sites = {
            s.site_id for s in submissions
            if s.status == SubmissionStatus.APPROVED.value
        }

        updated = round_.model_copy(update={
            "sites_submitted": len(submitted_sites),
            "sites_approved": len(approved_sites),
        })
        self._rounds[round_id] = updated

    def add_custom_validation_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule to the engine.

        Args:
            rule: The validation rule to add.
        """
        self._custom_rules.append(rule)
        logger.info("Custom validation rule '%s' added.", rule.rule_name)

    def get_round(self, round_id: str) -> CollectionRound:
        """Retrieve a collection round by ID.

        Args:
            round_id: The round ID.

        Returns:
            The CollectionRound.

        Raises:
            KeyError: If not found.
        """
        if round_id not in self._rounds:
            raise KeyError(f"Round '{round_id}' not found.")
        return self._rounds[round_id]

    def get_submission(self, submission_id: str) -> SiteSubmission:
        """Retrieve a submission by ID.

        Args:
            submission_id: The submission ID.

        Returns:
            The SiteSubmission.

        Raises:
            KeyError: If not found.
        """
        if submission_id not in self._submissions:
            raise KeyError(f"Submission '{submission_id}' not found.")
        return self._submissions[submission_id]

    def get_submissions_for_round(
        self, round_id: str,
    ) -> List[SiteSubmission]:
        """Get all submissions for a specific round.

        Args:
            round_id: The collection round ID.

        Returns:
            List of SiteSubmissions for the round.
        """
        return [
            s for s in self._submissions.values()
            if s.round_id == round_id
        ]

    def get_submissions_for_site(
        self, site_id: str,
    ) -> List[SiteSubmission]:
        """Get all submissions for a specific site.

        Args:
            site_id: The site ID.

        Returns:
            List of SiteSubmissions for the site.
        """
        return [
            s for s in self._submissions.values()
            if s.site_id == site_id
        ]
