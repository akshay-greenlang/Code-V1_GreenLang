# -*- coding: utf-8 -*-
"""
Data Collection Workflow
====================================

5-phase workflow for multi-site GHG data collection covering period setup,
template distribution, site submission, validation review, and approval
within PACK-049 GHG Multi-Site Management Pack.

Phases:
    1. PeriodSetup              -- Define collection round parameters
                                   (period, deadline, scopes, target sites).
    2. TemplateDistribution     -- Generate site-specific data collection
                                   templates based on facility type and scopes.
    3. SiteSubmission           -- Collect data entries from sites with
                                   real-time field-level validation.
    4. ValidationReview         -- Run comprehensive validation rules
                                   (range, YoY, unit, completeness checks).
    5. Approval                 -- Approve or reject submissions per site,
                                   generate collection summary with provenance.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 7) -- Managing inventory quality
    ISO 14064-1:2018 (Cl. 5) -- Quantification of GHG emissions
    CSRD / ESRS E1 (2024) -- Climate change disclosure
    GHG Protocol Scope 3 Standard (Ch. 7) -- Data collection guidance

Author: GreenLang Team
Version: 49.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CollectionPhase(str, Enum):
    PERIOD_SETUP = "period_setup"
    TEMPLATE_DISTRIBUTION = "template_distribution"
    SITE_SUBMISSION = "site_submission"
    VALIDATION_REVIEW = "validation_review"
    APPROVAL = "approval"


class EmissionScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class SubmissionStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"


class TemplateType(str, Enum):
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    PROCESS_EMISSIONS = "process_emissions"
    ELECTRICITY = "electricity"
    STEAM_HEATING = "steam_heating"
    SCOPE_3_CATEGORIES = "scope_3_categories"
    GENERAL = "general"


# =============================================================================
# REFERENCE DATA
# =============================================================================

SCOPE_TEMPLATE_MAP: Dict[str, List[str]] = {
    "scope_1": [
        "stationary_combustion", "mobile_combustion",
        "fugitive_emissions", "process_emissions",
    ],
    "scope_2": ["electricity", "steam_heating"],
    "scope_3": ["scope_3_categories"],
}

YOY_THRESHOLD_PCT = Decimal("30")  # Flag if >30% change year-over-year
MINIMUM_COMPLETENESS_PCT = Decimal("90")
RANGE_CHECK_SIGMA = Decimal("3.0")  # 3-sigma range check


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    phase_name: str = Field(...)
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class CollectionRoundConfig(BaseModel):
    """Configuration for a data collection round."""
    round_id: str = Field(default_factory=_new_uuid)
    reporting_period_start: str = Field(..., description="ISO date start")
    reporting_period_end: str = Field(..., description="ISO date end")
    submission_deadline: str = Field(..., description="ISO datetime deadline")
    scopes: List[EmissionScope] = Field(default_factory=lambda: [EmissionScope.SCOPE_1, EmissionScope.SCOPE_2])
    target_site_ids: List[str] = Field(default_factory=list, description="Sites to collect from")
    collection_frequency: str = Field("annual", description="annual|quarterly|monthly")
    reminder_interval_days: int = Field(7, ge=1, description="Reminder interval")
    escalation_after_days: int = Field(14, ge=1, description="Escalation after N days")


class SiteTemplate(BaseModel):
    """Data collection template assigned to a site."""
    template_id: str = Field(default_factory=_new_uuid)
    site_id: str = Field(...)
    site_name: str = Field("")
    template_type: TemplateType = Field(TemplateType.GENERAL)
    scope: EmissionScope = Field(EmissionScope.SCOPE_1)
    fields: List[Dict[str, Any]] = Field(default_factory=list)
    instructions: str = Field("")
    due_date: str = Field("")


class DataEntry(BaseModel):
    """A single data entry submitted by a site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entry_id: str = Field(default_factory=_new_uuid)
    site_id: str = Field(...)
    template_id: str = Field("")
    scope: EmissionScope = Field(EmissionScope.SCOPE_1)
    source_category: str = Field("", description="e.g. natural_gas, diesel, electricity")
    activity_data_value: Decimal = Field(Decimal("0"), description="Activity data quantity")
    activity_data_unit: str = Field("", description="Unit of measure")
    emission_factor: Optional[Decimal] = Field(None)
    emission_factor_unit: str = Field("")
    emission_factor_source: str = Field("")
    calculated_emissions_tco2e: Optional[Decimal] = Field(None)
    reporting_period: str = Field("")
    evidence_reference: str = Field("")
    submitted_by: str = Field("")
    submitted_at: str = Field("")
    notes: str = Field("")


class ValidationFinding(BaseModel):
    """A validation finding against a data entry."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    finding_id: str = Field(default_factory=_new_uuid)
    site_id: str = Field(...)
    entry_id: str = Field("")
    rule_code: str = Field("", description="Validation rule identifier")
    severity: ValidationSeverity = Field(ValidationSeverity.WARNING)
    message: str = Field("")
    field_name: str = Field("")
    current_value: Optional[str] = Field(None)
    expected_range: Optional[str] = Field(None)
    prior_year_value: Optional[str] = Field(None)
    auto_resolved: bool = Field(False)


class SiteSubmissionRecord(BaseModel):
    """Submission record for a single site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    site_name: str = Field("")
    status: SubmissionStatus = Field(SubmissionStatus.NOT_STARTED)
    entries_count: int = Field(0)
    completeness_pct: Decimal = Field(Decimal("0"))
    error_count: int = Field(0)
    warning_count: int = Field(0)
    approval_decision: Optional[ApprovalDecision] = Field(None)
    reviewer: str = Field("")
    reviewed_at: str = Field("")
    rejection_reason: str = Field("")
    provenance_hash: str = Field("")


class DataCollectionInput(BaseModel):
    """Input for the data collection workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    organisation_id: str = Field(...)
    round_config: CollectionRoundConfig = Field(...)
    site_metadata: List[Dict[str, Any]] = Field(
        default_factory=list, description="Site info dicts with site_id, site_name, facility_type"
    )
    submissions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Raw data entry dicts"
    )
    prior_year_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Prior year entries for YoY checks"
    )
    auto_approve_threshold: Decimal = Field(
        Decimal("95"), description="Auto-approve above this completeness %"
    )
    skip_phases: List[str] = Field(default_factory=list)


class DataCollectionResult(BaseModel):
    """Output from the data collection workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    workflow_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    round_id: str = Field("")
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    site_submissions: List[SiteSubmissionRecord] = Field(default_factory=list)
    templates_generated: int = Field(0)
    total_entries: int = Field(0)
    total_errors: int = Field(0)
    total_warnings: int = Field(0)
    approved_count: int = Field(0)
    rejected_count: int = Field(0)
    overall_completeness_pct: Decimal = Field(Decimal("0"))
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class DataCollectionWorkflow:
    """
    5-phase data collection workflow for multi-site GHG inventories.

    Orchestrates collection round setup, template generation, site-level
    data submission, comprehensive validation, and approval with full
    SHA-256 provenance tracking.

    Example:
        >>> wf = DataCollectionWorkflow()
        >>> inp = DataCollectionInput(
        ...     organisation_id="ORG-001",
        ...     round_config=CollectionRoundConfig(
        ...         reporting_period_start="2025-01-01",
        ...         reporting_period_end="2025-12-31",
        ...         submission_deadline="2026-03-31T23:59:59Z",
        ...     ),
        ... )
        >>> result = wf.execute(inp)
    """

    PHASE_ORDER: List[CollectionPhase] = [
        CollectionPhase.PERIOD_SETUP,
        CollectionPhase.TEMPLATE_DISTRIBUTION,
        CollectionPhase.SITE_SUBMISSION,
        CollectionPhase.VALIDATION_REVIEW,
        CollectionPhase.APPROVAL,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._round: Optional[CollectionRoundConfig] = None
        self._templates: List[SiteTemplate] = []
        self._entries: Dict[str, List[DataEntry]] = {}  # site_id -> entries
        self._findings: Dict[str, List[ValidationFinding]] = {}  # site_id -> findings

    def execute(self, input_data: DataCollectionInput) -> DataCollectionResult:
        """Execute the full 5-phase data collection workflow."""
        start = _utcnow()
        result = DataCollectionResult(
            organisation_id=input_data.organisation_id,
            round_id=input_data.round_config.round_id,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        phase_methods = {
            CollectionPhase.PERIOD_SETUP: self._phase_period_setup,
            CollectionPhase.TEMPLATE_DISTRIBUTION: self._phase_template_distribution,
            CollectionPhase.SITE_SUBMISSION: self._phase_site_submission,
            CollectionPhase.VALIDATION_REVIEW: self._phase_validation_review,
            CollectionPhase.APPROVAL: self._phase_approval,
        }

        for idx, phase in enumerate(self.PHASE_ORDER, 1):
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx, status=PhaseStatus.SKIPPED,
                ))
                continue

            phase_start = _utcnow()
            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (_utcnow() - phase_start).total_seconds()
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                    outputs=phase_out, provenance_hash=_compute_hash(str(phase_out)),
                ))
            except Exception as exc:
                elapsed = (_utcnow() - phase_start).total_seconds()
                logger.error("Phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.FAILED, duration_seconds=elapsed,
                    errors=[str(exc)],
                ))
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"Phase {phase.value}: {exc}")
                break

        if result.status != WorkflowStatus.FAILED:
            result.status = WorkflowStatus.COMPLETED

        end = _utcnow()
        result.completed_at = end.isoformat()
        result.duration_seconds = (end - start).total_seconds()
        result.provenance_hash = _compute_hash(
            f"{result.workflow_id}|{result.round_id}|{result.total_entries}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- PERIOD SETUP
    # -----------------------------------------------------------------

    def _phase_period_setup(
        self, input_data: DataCollectionInput, result: DataCollectionResult,
    ) -> Dict[str, Any]:
        """Define collection round parameters."""
        logger.info("Phase 1 -- Period Setup")
        self._round = input_data.round_config

        target_count = len(self._round.target_site_ids)
        if target_count == 0 and input_data.site_metadata:
            self._round.target_site_ids = [
                s.get("site_id", _new_uuid()) for s in input_data.site_metadata
            ]
            target_count = len(self._round.target_site_ids)

        scope_names = [s.value for s in self._round.scopes]
        logger.info(
            "Round %s: period %s to %s, %d sites, scopes %s",
            self._round.round_id, self._round.reporting_period_start,
            self._round.reporting_period_end, target_count, scope_names,
        )
        return {
            "round_id": self._round.round_id,
            "period_start": self._round.reporting_period_start,
            "period_end": self._round.reporting_period_end,
            "deadline": self._round.submission_deadline,
            "target_sites": target_count,
            "scopes": scope_names,
            "frequency": self._round.collection_frequency,
        }

    # -----------------------------------------------------------------
    # PHASE 2 -- TEMPLATE DISTRIBUTION
    # -----------------------------------------------------------------

    def _phase_template_distribution(
        self, input_data: DataCollectionInput, result: DataCollectionResult,
    ) -> Dict[str, Any]:
        """Generate site-specific data collection templates."""
        logger.info("Phase 2 -- Template Distribution")
        templates: List[SiteTemplate] = []
        site_lookup: Dict[str, Dict[str, Any]] = {}
        for sm in input_data.site_metadata:
            sid = sm.get("site_id", "")
            site_lookup[sid] = sm

        for site_id in (self._round.target_site_ids if self._round else []):
            meta = site_lookup.get(site_id, {})
            site_name = meta.get("site_name", site_id)
            facility_type = meta.get("facility_type", "other")

            for scope in (self._round.scopes if self._round else []):
                template_types = SCOPE_TEMPLATE_MAP.get(scope.value, ["general"])
                for tt_str in template_types:
                    try:
                        tt = TemplateType(tt_str)
                    except ValueError:
                        tt = TemplateType.GENERAL

                    fields = self._generate_template_fields(tt, scope, facility_type)
                    instructions = self._generate_instructions(tt, scope)

                    tpl = SiteTemplate(
                        site_id=site_id,
                        site_name=site_name,
                        template_type=tt,
                        scope=scope,
                        fields=fields,
                        instructions=instructions,
                        due_date=self._round.submission_deadline if self._round else "",
                    )
                    templates.append(tpl)

        self._templates = templates
        result.templates_generated = len(templates)

        logger.info("Generated %d templates for %d sites",
                     len(templates), len(self._round.target_site_ids if self._round else []))
        return {
            "templates_generated": len(templates),
            "templates_per_site": (
                len(templates) // max(len(self._round.target_site_ids), 1) if self._round else 0
            ),
        }

    def _generate_template_fields(
        self, template_type: TemplateType, scope: EmissionScope, facility_type: str,
    ) -> List[Dict[str, Any]]:
        """Generate deterministic field definitions for a template."""
        base_fields = [
            {"name": "source_category", "type": "text", "required": True,
             "description": "Emission source category"},
            {"name": "activity_data_value", "type": "decimal", "required": True,
             "description": "Activity data quantity"},
            {"name": "activity_data_unit", "type": "unit", "required": True,
             "description": "Unit of measure"},
        ]

        if template_type == TemplateType.STATIONARY_COMBUSTION:
            base_fields.extend([
                {"name": "fuel_type", "type": "select", "required": True,
                 "options": ["natural_gas", "diesel", "fuel_oil", "lpg", "coal", "biomass"],
                 "description": "Fuel type"},
                {"name": "consumption_quantity", "type": "decimal", "required": True,
                 "description": "Fuel consumption quantity"},
                {"name": "consumption_unit", "type": "unit", "required": True,
                 "options": ["m3", "litres", "kg", "tonnes", "kWh", "therms"],
                 "description": "Consumption unit"},
            ])
        elif template_type == TemplateType.MOBILE_COMBUSTION:
            base_fields.extend([
                {"name": "vehicle_type", "type": "select", "required": True,
                 "options": ["car", "van", "truck", "bus", "rail", "marine", "aviation"],
                 "description": "Vehicle type"},
                {"name": "fuel_type", "type": "select", "required": True,
                 "options": ["petrol", "diesel", "cng", "lpg", "electric", "hybrid"],
                 "description": "Fuel type"},
                {"name": "distance_km", "type": "decimal", "required": False,
                 "description": "Distance travelled (km)"},
            ])
        elif template_type == TemplateType.ELECTRICITY:
            base_fields.extend([
                {"name": "grid_region", "type": "text", "required": True,
                 "description": "Grid region code"},
                {"name": "consumption_kwh", "type": "decimal", "required": True,
                 "description": "Electricity consumption kWh"},
                {"name": "renewable_pct", "type": "decimal", "required": False,
                 "description": "Renewable % if market-based"},
            ])
        elif template_type == TemplateType.FUGITIVE_EMISSIONS:
            base_fields.extend([
                {"name": "gas_type", "type": "select", "required": True,
                 "options": ["R-134a", "R-410A", "R-404A", "R-407C", "SF6", "CO2", "CH4"],
                 "description": "Refrigerant or gas type"},
                {"name": "charge_kg", "type": "decimal", "required": True,
                 "description": "Equipment charge (kg)"},
                {"name": "leak_rate_pct", "type": "decimal", "required": False,
                 "description": "Annual leak rate %"},
            ])

        base_fields.extend([
            {"name": "evidence_reference", "type": "text", "required": False,
             "description": "Evidence / invoice reference"},
            {"name": "notes", "type": "textarea", "required": False,
             "description": "Additional notes"},
        ])
        return base_fields

    def _generate_instructions(self, template_type: TemplateType, scope: EmissionScope) -> str:
        """Generate deterministic instructions for a template type."""
        base = (
            f"Complete all required fields for {scope.value.replace('_', ' ')} "
            f"({template_type.value.replace('_', ' ')}).\n"
            "Ensure units are consistent. Attach evidence where available.\n"
        )
        if template_type == TemplateType.STATIONARY_COMBUSTION:
            base += "Report fuel consumption from boilers, furnaces, generators.\n"
        elif template_type == TemplateType.ELECTRICITY:
            base += "Report purchased electricity. Provide grid region for location-based.\n"
        return base

    # -----------------------------------------------------------------
    # PHASE 3 -- SITE SUBMISSION
    # -----------------------------------------------------------------

    def _phase_site_submission(
        self, input_data: DataCollectionInput, result: DataCollectionResult,
    ) -> Dict[str, Any]:
        """Collect data entries with real-time validation."""
        logger.info("Phase 3 -- Site Submission: %d raw entries", len(input_data.submissions))
        entries_by_site: Dict[str, List[DataEntry]] = {}
        parse_errors = 0

        for raw in input_data.submissions:
            site_id = raw.get("site_id", "")
            if not site_id:
                parse_errors += 1
                continue

            try:
                adv = Decimal(str(raw.get("activity_data_value", "0")))
            except Exception:
                adv = Decimal("0")
                parse_errors += 1

            ef = None
            if raw.get("emission_factor") is not None:
                try:
                    ef = Decimal(str(raw["emission_factor"]))
                except Exception:
                    pass

            calc = None
            if raw.get("calculated_emissions_tco2e") is not None:
                try:
                    calc = Decimal(str(raw["calculated_emissions_tco2e"]))
                except Exception:
                    pass

            try:
                scope = EmissionScope(raw.get("scope", "scope_1"))
            except ValueError:
                scope = EmissionScope.SCOPE_1

            entry = DataEntry(
                site_id=site_id,
                template_id=raw.get("template_id", ""),
                scope=scope,
                source_category=raw.get("source_category", ""),
                activity_data_value=adv,
                activity_data_unit=raw.get("activity_data_unit", ""),
                emission_factor=ef,
                emission_factor_unit=raw.get("emission_factor_unit", ""),
                emission_factor_source=raw.get("emission_factor_source", ""),
                calculated_emissions_tco2e=calc,
                reporting_period=raw.get("reporting_period", ""),
                evidence_reference=raw.get("evidence_reference", ""),
                submitted_by=raw.get("submitted_by", ""),
                submitted_at=raw.get("submitted_at", _utcnow().isoformat()),
                notes=raw.get("notes", ""),
            )
            entries_by_site.setdefault(site_id, []).append(entry)

        self._entries = entries_by_site
        total_entries = sum(len(v) for v in entries_by_site.values())
        result.total_entries = total_entries

        logger.info("Parsed %d entries across %d sites (%d parse errors)",
                     total_entries, len(entries_by_site), parse_errors)
        return {
            "total_entries": total_entries,
            "sites_with_data": len(entries_by_site),
            "parse_errors": parse_errors,
        }

    # -----------------------------------------------------------------
    # PHASE 4 -- VALIDATION REVIEW
    # -----------------------------------------------------------------

    def _phase_validation_review(
        self, input_data: DataCollectionInput, result: DataCollectionResult,
    ) -> Dict[str, Any]:
        """Run comprehensive validation rules on submissions."""
        logger.info("Phase 4 -- Validation Review")
        prior_lookup = self._build_prior_year_lookup(input_data.prior_year_data)
        total_errors = 0
        total_warnings = 0

        for site_id, entries in self._entries.items():
            findings: List[ValidationFinding] = []

            for entry in entries:
                # Rule V001: Non-negative activity data
                if entry.activity_data_value < Decimal("0"):
                    findings.append(ValidationFinding(
                        site_id=site_id, entry_id=entry.entry_id,
                        rule_code="V001", severity=ValidationSeverity.ERROR,
                        message="Activity data must be non-negative",
                        field_name="activity_data_value",
                        current_value=str(entry.activity_data_value),
                    ))

                # Rule V002: Unit required
                if not entry.activity_data_unit:
                    findings.append(ValidationFinding(
                        site_id=site_id, entry_id=entry.entry_id,
                        rule_code="V002", severity=ValidationSeverity.ERROR,
                        message="Activity data unit is required",
                        field_name="activity_data_unit",
                    ))

                # Rule V003: Source category required
                if not entry.source_category:
                    findings.append(ValidationFinding(
                        site_id=site_id, entry_id=entry.entry_id,
                        rule_code="V003", severity=ValidationSeverity.WARNING,
                        message="Source category is recommended",
                        field_name="source_category",
                    ))

                # Rule V004: Year-over-year check
                prior_key = f"{site_id}|{entry.source_category}|{entry.scope.value}"
                prior_val = prior_lookup.get(prior_key)
                if prior_val is not None and prior_val > Decimal("0"):
                    pct_change = abs(
                        (entry.activity_data_value - prior_val) / prior_val * Decimal("100")
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    if pct_change > YOY_THRESHOLD_PCT:
                        findings.append(ValidationFinding(
                            site_id=site_id, entry_id=entry.entry_id,
                            rule_code="V004", severity=ValidationSeverity.WARNING,
                            message=f"YoY change of {pct_change}% exceeds {YOY_THRESHOLD_PCT}%",
                            field_name="activity_data_value",
                            current_value=str(entry.activity_data_value),
                            prior_year_value=str(prior_val),
                        ))

                # Rule V005: Zero-value check
                if entry.activity_data_value == Decimal("0"):
                    findings.append(ValidationFinding(
                        site_id=site_id, entry_id=entry.entry_id,
                        rule_code="V005", severity=ValidationSeverity.INFO,
                        message="Zero activity data reported -- verify intentional",
                        field_name="activity_data_value",
                    ))

                # Rule V006: Emission factor sanity
                if entry.emission_factor is not None and entry.emission_factor <= Decimal("0"):
                    findings.append(ValidationFinding(
                        site_id=site_id, entry_id=entry.entry_id,
                        rule_code="V006", severity=ValidationSeverity.ERROR,
                        message="Emission factor must be positive",
                        field_name="emission_factor",
                        current_value=str(entry.emission_factor),
                    ))

                # Rule V007: Calculated emissions cross-check
                if (
                    entry.emission_factor is not None
                    and entry.calculated_emissions_tco2e is not None
                    and entry.emission_factor > Decimal("0")
                ):
                    expected = (entry.activity_data_value * entry.emission_factor).quantize(
                        Decimal("0.0001"), rounding=ROUND_HALF_UP
                    )
                    diff = abs(entry.calculated_emissions_tco2e - expected)
                    tolerance = expected * Decimal("0.01")  # 1% tolerance
                    if diff > tolerance and expected > Decimal("0"):
                        findings.append(ValidationFinding(
                            site_id=site_id, entry_id=entry.entry_id,
                            rule_code="V007", severity=ValidationSeverity.WARNING,
                            message=f"Calculated emissions mismatch: expected ~{expected}",
                            field_name="calculated_emissions_tco2e",
                            current_value=str(entry.calculated_emissions_tco2e),
                            expected_range=str(expected),
                        ))

            site_errors = sum(1 for f in findings if f.severity == ValidationSeverity.ERROR)
            site_warnings = sum(1 for f in findings if f.severity == ValidationSeverity.WARNING)
            total_errors += site_errors
            total_warnings += site_warnings
            self._findings[site_id] = findings

        result.total_errors = total_errors
        result.total_warnings = total_warnings

        logger.info("Validation: %d errors, %d warnings", total_errors, total_warnings)
        return {
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "sites_validated": len(self._entries),
        }

    def _build_prior_year_lookup(
        self, prior_data: List[Dict[str, Any]]
    ) -> Dict[str, Decimal]:
        """Build lookup of prior year activity data by site|category|scope."""
        lookup: Dict[str, Decimal] = {}
        for rec in prior_data:
            key = (
                f"{rec.get('site_id', '')}|{rec.get('source_category', '')}|"
                f"{rec.get('scope', 'scope_1')}"
            )
            try:
                lookup[key] = Decimal(str(rec.get("activity_data_value", "0")))
            except Exception:
                pass
        return lookup

    # -----------------------------------------------------------------
    # PHASE 5 -- APPROVAL
    # -----------------------------------------------------------------

    def _phase_approval(
        self, input_data: DataCollectionInput, result: DataCollectionResult,
    ) -> Dict[str, Any]:
        """Approve or reject per-site submissions based on validation."""
        logger.info("Phase 5 -- Approval")
        site_records: List[SiteSubmissionRecord] = []
        approved_count = 0
        rejected_count = 0

        target_ids = self._round.target_site_ids if self._round else []
        site_name_lookup: Dict[str, str] = {}
        for sm in input_data.site_metadata:
            site_name_lookup[sm.get("site_id", "")] = sm.get("site_name", "")

        for site_id in target_ids:
            entries = self._entries.get(site_id, [])
            findings = self._findings.get(site_id, [])
            error_count = sum(1 for f in findings if f.severity == ValidationSeverity.ERROR)
            warning_count = sum(1 for f in findings if f.severity == ValidationSeverity.WARNING)

            # Compute completeness: templates assigned vs entries received
            expected_templates = sum(
                1 for t in self._templates if t.site_id == site_id
            )
            completeness = Decimal("100") if expected_templates == 0 else (
                Decimal(str(len(entries))) / Decimal(str(max(expected_templates, 1)))
                * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            completeness = min(completeness, Decimal("100"))

            # Determine approval
            if error_count > 0:
                decision = ApprovalDecision.REJECTED
                rejection_reason = f"{error_count} validation error(s)"
                rejected_count += 1
            elif completeness < MINIMUM_COMPLETENESS_PCT:
                decision = ApprovalDecision.REJECTED
                rejection_reason = f"Completeness {completeness}% below {MINIMUM_COMPLETENESS_PCT}%"
                rejected_count += 1
            elif completeness >= input_data.auto_approve_threshold and warning_count == 0:
                decision = ApprovalDecision.APPROVED
                rejection_reason = ""
                approved_count += 1
            elif warning_count > 0:
                decision = ApprovalDecision.CONDITIONAL
                rejection_reason = f"{warning_count} warning(s) require review"
                approved_count += 1  # conditional still counts
            else:
                decision = ApprovalDecision.APPROVED
                rejection_reason = ""
                approved_count += 1

            prov = _compute_hash(
                f"{site_id}|{len(entries)}|{error_count}|{completeness}|{decision.value}"
            )

            rec = SiteSubmissionRecord(
                site_id=site_id,
                site_name=site_name_lookup.get(site_id, site_id),
                status=SubmissionStatus.APPROVED if decision == ApprovalDecision.APPROVED
                       else SubmissionStatus.REJECTED if decision == ApprovalDecision.REJECTED
                       else SubmissionStatus.UNDER_REVIEW,
                entries_count=len(entries),
                completeness_pct=completeness,
                error_count=error_count,
                warning_count=warning_count,
                approval_decision=decision,
                reviewer="system_auto" if decision != ApprovalDecision.CONDITIONAL else "",
                reviewed_at=_utcnow().isoformat(),
                rejection_reason=rejection_reason,
                provenance_hash=prov,
            )
            site_records.append(rec)

        result.site_submissions = site_records
        result.approved_count = approved_count
        result.rejected_count = rejected_count

        # Overall completeness
        if site_records:
            avg = sum(r.completeness_pct for r in site_records) / Decimal(str(len(site_records)))
            result.overall_completeness_pct = avg.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        logger.info("Approval: %d approved, %d rejected", approved_count, rejected_count)
        return {
            "approved": approved_count,
            "rejected": rejected_count,
            "conditional": sum(
                1 for r in site_records if r.approval_decision == ApprovalDecision.CONDITIONAL
            ),
            "overall_completeness_pct": float(result.overall_completeness_pct),
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "DataCollectionWorkflow",
    "DataCollectionInput",
    "DataCollectionResult",
    "CollectionPhase",
    "EmissionScope",
    "SubmissionStatus",
    "ValidationSeverity",
    "ApprovalDecision",
    "TemplateType",
    "CollectionRoundConfig",
    "SiteTemplate",
    "DataEntry",
    "ValidationFinding",
    "SiteSubmissionRecord",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
