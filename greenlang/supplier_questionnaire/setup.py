# -*- coding: utf-8 -*-
"""
Supplier Questionnaire Processor Service Setup - AGENT-DATA-008

Provides ``configure_supplier_questionnaire(app)`` which wires up the
Supplier Questionnaire Processor SDK (template builder, distribution engine,
response collector, validation engine, scoring engine, follow-up manager,
analytics engine, provenance tracker) and mounts the REST API.

Also exposes ``get_supplier_questionnaire(app)`` for programmatic access
and the ``SupplierQuestionnaireService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.supplier_questionnaire.setup import configure_supplier_questionnaire
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_supplier_questionnaire(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.supplier_questionnaire.config import (
    SupplierQuestionnaireConfig,
    get_config,
)
from greenlang.supplier_questionnaire.metrics import (
    PROMETHEUS_AVAILABLE,
    record_template,
    record_distribution,
    record_response,
    record_validation,
    record_score,
    record_followup,
    update_response_rate,
    record_processing_duration,
    update_active_campaigns,
    update_pending_responses,
    record_processing_error,
    record_data_quality,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# Lightweight Pydantic models used by the facade
# ===================================================================


class QuestionnaireTemplate(BaseModel):
    """Questionnaire template definition.

    Attributes:
        template_id: Unique template identifier.
        name: Template display name.
        framework: Questionnaire framework (cdp, ecovadis, gri, custom, etc.).
        version: Template version string.
        description: Template description.
        sections: Ordered list of section definitions.
        questions: Total number of questions across all sections.
        language: ISO 639-1 language code.
        tags: Classification tags.
        status: Template status (draft, active, archived).
        created_by: User who created the template.
        provenance_hash: SHA-256 provenance hash.
        created_at: Timestamp of creation.
        updated_at: Timestamp of last update.
    """
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    framework: str = Field(default="custom")
    version: str = Field(default="1.0")
    description: str = Field(default="")
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    questions: int = Field(default=0)
    language: str = Field(default="en")
    tags: List[str] = Field(default_factory=list)
    status: str = Field(default="draft")
    created_by: str = Field(default="system")
    provenance_hash: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class Distribution(BaseModel):
    """Questionnaire distribution record.

    Attributes:
        distribution_id: Unique distribution identifier.
        template_id: Source template identifier.
        campaign_id: Campaign this distribution belongs to.
        supplier_id: Target supplier identifier.
        supplier_name: Target supplier display name.
        supplier_email: Supplier contact email.
        channel: Distribution channel (email, portal, api, bulk).
        status: Distribution status (pending, sent, delivered, bounced, failed).
        deadline: Response deadline (ISO 8601).
        reminder_count: Number of reminders sent.
        provenance_hash: SHA-256 provenance hash.
        distributed_at: Timestamp of distribution.
    """
    distribution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template_id: str = Field(default="")
    campaign_id: str = Field(default="")
    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    supplier_email: str = Field(default="")
    channel: str = Field(default="email")
    status: str = Field(default="pending")
    deadline: str = Field(default="")
    reminder_count: int = Field(default=0)
    provenance_hash: str = Field(default="")
    distributed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class QuestionnaireResponse(BaseModel):
    """Supplier questionnaire response record.

    Attributes:
        response_id: Unique response identifier.
        distribution_id: Linked distribution identifier.
        template_id: Source template identifier.
        supplier_id: Responding supplier identifier.
        supplier_name: Responding supplier display name.
        answers: Answer data keyed by question_id.
        completion_pct: Response completion percentage.
        status: Response status (draft, submitted, finalized, rejected).
        channel: Response submission channel.
        evidence_files: Attached evidence file references.
        submitted_at: Timestamp of submission.
        finalized_at: Timestamp of finalization.
        provenance_hash: SHA-256 provenance hash.
    """
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    distribution_id: str = Field(default="")
    template_id: str = Field(default="")
    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    answers: Dict[str, Any] = Field(default_factory=dict)
    completion_pct: float = Field(default=0.0)
    status: str = Field(default="draft")
    channel: str = Field(default="portal")
    evidence_files: List[str] = Field(default_factory=list)
    submitted_at: Optional[str] = Field(default=None)
    finalized_at: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")


class ValidationResult(BaseModel):
    """Result of a questionnaire response validation.

    Attributes:
        validation_id: Unique validation identifier.
        response_id: Validated response identifier.
        is_valid: Overall validation result.
        completion_pct: Response completion percentage.
        errors: List of validation error descriptions.
        warnings: List of validation warning descriptions.
        checks_passed: Number of validation checks that passed.
        checks_failed: Number of validation checks that failed.
        checks_warned: Number of checks that produced warnings.
        level: Validation level applied (completeness, consistency, evidence, cross_field).
        provenance_hash: SHA-256 provenance hash.
        validated_at: Timestamp of validation.
    """
    validation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    response_id: str = Field(default="")
    is_valid: bool = Field(default=False)
    completion_pct: float = Field(default=0.0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    checks_passed: int = Field(default=0)
    checks_failed: int = Field(default=0)
    checks_warned: int = Field(default=0)
    level: str = Field(default="completeness")
    provenance_hash: str = Field(default="")
    validated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class ScoringResult(BaseModel):
    """Result of scoring a questionnaire response.

    Attributes:
        score_id: Unique score identifier.
        response_id: Scored response identifier.
        supplier_id: Scored supplier identifier.
        framework: Scoring framework used.
        total_score: Overall score (0-100).
        tier: Performance tier (leader, advanced, developing, lagging).
        section_scores: Scores per section.
        category_scores: Scores per category.
        benchmark_percentile: Percentile rank vs. peer suppliers.
        provenance_hash: SHA-256 provenance hash.
        scored_at: Timestamp of scoring.
    """
    score_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    response_id: str = Field(default="")
    supplier_id: str = Field(default="")
    framework: str = Field(default="custom")
    total_score: float = Field(default=0.0)
    tier: str = Field(default="lagging")
    section_scores: Dict[str, float] = Field(default_factory=dict)
    category_scores: Dict[str, float] = Field(default_factory=dict)
    benchmark_percentile: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    scored_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class FollowUpAction(BaseModel):
    """Follow-up action record for a distribution.

    Attributes:
        action_id: Unique action identifier.
        distribution_id: Target distribution identifier.
        campaign_id: Owning campaign identifier.
        supplier_id: Target supplier identifier.
        action_type: Follow-up type (reminder, escalation, deadline_extension).
        status: Action status (scheduled, sent, acknowledged, expired).
        scheduled_at: Scheduled execution time.
        executed_at: Actual execution time.
        message: Follow-up message content.
        provenance_hash: SHA-256 provenance hash.
    """
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    distribution_id: str = Field(default="")
    campaign_id: str = Field(default="")
    supplier_id: str = Field(default="")
    action_type: str = Field(default="reminder")
    status: str = Field(default="scheduled")
    scheduled_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    executed_at: Optional[str] = Field(default=None)
    message: str = Field(default="")
    provenance_hash: str = Field(default="")


class CampaignAnalytics(BaseModel):
    """Analytics summary for a questionnaire campaign.

    Attributes:
        campaign_id: Campaign identifier.
        total_distributed: Total questionnaires distributed.
        total_responded: Total responses received.
        total_finalized: Total responses finalized.
        response_rate_pct: Response rate percentage.
        avg_completion_pct: Average completion percentage.
        avg_score: Average score across responses.
        score_distribution: Count of suppliers per performance tier.
        compliance_gaps: List of identified compliance gaps.
        provenance_hash: SHA-256 provenance hash.
        generated_at: Timestamp of analytics generation.
    """
    campaign_id: str = Field(default="")
    total_distributed: int = Field(default=0)
    total_responded: int = Field(default=0)
    total_finalized: int = Field(default=0)
    response_rate_pct: float = Field(default=0.0)
    avg_completion_pct: float = Field(default=0.0)
    avg_score: float = Field(default=0.0)
    score_distribution: Dict[str, int] = Field(default_factory=dict)
    compliance_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class QuestionnaireStatistics(BaseModel):
    """Aggregate statistics for the supplier questionnaire service.

    Attributes:
        total_templates: Total templates managed.
        active_templates: Currently active templates.
        total_distributions: Total distributions sent.
        total_responses: Total responses received.
        total_finalized: Total responses finalized.
        total_validations: Total validations performed.
        total_scores: Total scores calculated.
        total_followups: Total follow-up actions triggered.
        total_campaigns: Total campaigns created.
        active_campaigns: Currently active campaigns.
        avg_response_rate_pct: Average response rate across campaigns.
        avg_score: Average score across all scored responses.
    """
    total_templates: int = Field(default=0)
    active_templates: int = Field(default=0)
    total_distributions: int = Field(default=0)
    total_responses: int = Field(default=0)
    total_finalized: int = Field(default=0)
    total_validations: int = Field(default=0)
    total_scores: int = Field(default=0)
    total_followups: int = Field(default=0)
    total_campaigns: int = Field(default=0)
    active_campaigns: int = Field(default=0)
    avg_response_rate_pct: float = Field(default=0.0)
    avg_score: float = Field(default=0.0)


# ===================================================================
# Provenance helper
# ===================================================================


class _ProvenanceTracker:
    """Minimal provenance tracker recording SHA-256 audit entries.

    Attributes:
        entries: List of provenance entries.
        entry_count: Number of entries recorded.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self.entry_count: int = 0

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
    ) -> str:
        """Record a provenance entry and return its hash.

        Args:
            entity_type: Type of entity (template, distribution, response, score, etc.).
            entity_id: Entity identifier.
            action: Action performed (create, distribute, submit, validate, score, etc.).
            data_hash: SHA-256 hash of associated data.
            user_id: User or system that performed the action.

        Returns:
            SHA-256 hash of the provenance entry itself.
        """
        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entry_hash = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()
        ).hexdigest()
        entry["entry_hash"] = entry_hash
        self._entries.append(entry)
        self.entry_count += 1
        return entry_hash


# ===================================================================
# SupplierQuestionnaireService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["SupplierQuestionnaireService"] = None


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class SupplierQuestionnaireService:
    """Unified facade over the Supplier Questionnaire Processor SDK.

    Aggregates all processor engines (template builder, distribution engine,
    response collector, validation engine, scoring engine, follow-up manager,
    analytics engine, provenance tracker) through a single entry point with
    convenience methods for common operations.

    Each method records provenance and updates self-monitoring metrics.

    Attributes:
        config: SupplierQuestionnaireConfig instance.
        provenance: _ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = SupplierQuestionnaireService()
        >>> template = service.create_template(
        ...     name="CDP Climate 2025", framework="cdp",
        ...     sections=[{"name": "Governance", "questions": [...]}],
        ... )
        >>> print(template.template_id, template.status)
    """

    def __init__(
        self,
        config: Optional[SupplierQuestionnaireConfig] = None,
    ) -> None:
        """Initialize the Supplier Questionnaire Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - TemplateBuilder
        - DistributionEngine
        - ResponseCollector
        - ValidationEngine
        - ScoringEngine
        - FollowUpManager
        - AnalyticsEngine

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Provenance tracker
        self.provenance = _ProvenanceTracker()

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._template_builder: Any = None
        self._distribution_engine: Any = None
        self._response_collector: Any = None
        self._validation_engine: Any = None
        self._scoring_engine: Any = None
        self._followup_manager: Any = None
        self._analytics_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._templates: Dict[str, QuestionnaireTemplate] = {}
        self._distributions: Dict[str, Distribution] = {}
        self._responses: Dict[str, QuestionnaireResponse] = {}
        self._validations: Dict[str, ValidationResult] = {}
        self._scores: Dict[str, ScoringResult] = {}
        self._followups: Dict[str, FollowUpAction] = {}
        self._campaigns: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._stats = QuestionnaireStatistics()
        self._started = False

        logger.info("SupplierQuestionnaireService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def template_builder(self) -> Any:
        """Get the TemplateBuilder engine instance."""
        return self._template_builder

    @property
    def distribution_engine(self) -> Any:
        """Get the DistributionEngine engine instance."""
        return self._distribution_engine

    @property
    def response_collector(self) -> Any:
        """Get the ResponseCollector engine instance."""
        return self._response_collector

    @property
    def validation_engine(self) -> Any:
        """Get the ValidationEngine engine instance."""
        return self._validation_engine

    @property
    def scoring_engine(self) -> Any:
        """Get the ScoringEngine engine instance."""
        return self._scoring_engine

    @property
    def followup_manager(self) -> Any:
        """Get the FollowUpManager engine instance."""
        return self._followup_manager

    @property
    def analytics_engine(self) -> Any:
        """Get the AnalyticsEngine engine instance."""
        return self._analytics_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        try:
            from greenlang.supplier_questionnaire.template_builder import TemplateBuilder
            self._template_builder = TemplateBuilder(self.config)
        except ImportError:
            logger.warning("TemplateBuilder not available; using stub")

        try:
            from greenlang.supplier_questionnaire.distribution_engine import DistributionEngine
            self._distribution_engine = DistributionEngine(self.config)
        except ImportError:
            logger.warning("DistributionEngine not available; using stub")

        try:
            from greenlang.supplier_questionnaire.response_collector import ResponseCollector
            self._response_collector = ResponseCollector(self.config)
        except ImportError:
            logger.warning("ResponseCollector not available; using stub")

        try:
            from greenlang.supplier_questionnaire.validation_engine import ValidationEngine
            self._validation_engine = ValidationEngine(self.config)
        except ImportError:
            logger.warning("ValidationEngine not available; using stub")

        try:
            from greenlang.supplier_questionnaire.scoring_engine import ScoringEngine
            self._scoring_engine = ScoringEngine(self.config)
        except ImportError:
            logger.warning("ScoringEngine not available; using stub")

        try:
            from greenlang.supplier_questionnaire.followup_manager import FollowUpManager
            self._followup_manager = FollowUpManager(self.config)
        except ImportError:
            logger.warning("FollowUpManager not available; using stub")

        try:
            from greenlang.supplier_questionnaire.analytics_engine import AnalyticsEngine
            self._analytics_engine = AnalyticsEngine(self.config)
        except ImportError:
            logger.warning("AnalyticsEngine not available; using stub")

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    def create_template(
        self,
        name: str,
        framework: str = "custom",
        version: str = "1.0",
        description: str = "",
        sections: Optional[List[Dict[str, Any]]] = None,
        language: str = "en",
        tags: Optional[List[str]] = None,
        created_by: str = "system",
    ) -> QuestionnaireTemplate:
        """Create a new questionnaire template.

        Args:
            name: Template display name.
            framework: Questionnaire framework (cdp, ecovadis, gri, custom).
            version: Template version string.
            description: Template description.
            sections: Ordered list of section definitions.
            language: ISO 639-1 language code.
            tags: Classification tags.
            created_by: User who created the template.

        Returns:
            QuestionnaireTemplate with registration details.

        Raises:
            ValueError: If name is empty.
        """
        start_time = time.time()

        if not name.strip():
            raise ValueError("Template name must not be empty")

        # Count total questions across sections
        section_list = sections or []
        total_questions = sum(
            len(s.get("questions", [])) for s in section_list
        )

        template = QuestionnaireTemplate(
            name=name,
            framework=framework,
            version=version,
            description=description,
            sections=section_list,
            questions=total_questions,
            language=language,
            tags=tags or [],
            status="draft",
            created_by=created_by,
        )

        # Compute provenance hash
        template.provenance_hash = _compute_hash(template)

        # Store template
        self._templates[template.template_id] = template

        # Record metrics
        record_template(framework, "created")
        record_processing_duration("create_template", time.time() - start_time)

        # Record provenance
        self.provenance.record(
            entity_type="template",
            entity_id=template.template_id,
            action="create",
            data_hash=template.provenance_hash,
            user_id=created_by,
        )

        # Update statistics
        self._stats.total_templates += 1
        self._stats.active_templates = sum(
            1 for t in self._templates.values() if t.status in ("draft", "active")
        )

        logger.info(
            "Created template %s (%s, framework=%s, %d questions)",
            template.template_id, name, framework, total_questions,
        )
        return template

    def get_template(self, template_id: str) -> Optional[QuestionnaireTemplate]:
        """Get a questionnaire template by ID.

        Args:
            template_id: Template identifier.

        Returns:
            QuestionnaireTemplate or None if not found.
        """
        return self._templates.get(template_id)

    def list_templates(
        self,
        framework: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[QuestionnaireTemplate]:
        """List questionnaire templates with optional filters.

        Args:
            framework: Optional framework filter.
            status: Optional status filter.
            limit: Maximum number of templates to return.
            offset: Number of templates to skip.

        Returns:
            List of QuestionnaireTemplate instances.
        """
        templates = list(self._templates.values())

        if framework is not None:
            templates = [t for t in templates if t.framework == framework]
        if status is not None:
            templates = [t for t in templates if t.status == status]

        return templates[offset:offset + limit]

    def update_template(
        self,
        template_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> QuestionnaireTemplate:
        """Update an existing questionnaire template.

        Args:
            template_id: Template identifier.
            name: New template name (optional).
            description: New description (optional).
            sections: New sections (optional).
            status: New status (optional).
            tags: New tags (optional).

        Returns:
            Updated QuestionnaireTemplate.

        Raises:
            ValueError: If template not found.
        """
        start_time = time.time()

        template = self._templates.get(template_id)
        if template is None:
            raise ValueError(f"Template {template_id} not found")

        if name is not None:
            template.name = name
        if description is not None:
            template.description = description
        if sections is not None:
            template.sections = sections
            template.questions = sum(
                len(s.get("questions", [])) for s in sections
            )
        if status is not None:
            template.status = status
        if tags is not None:
            template.tags = tags

        template.updated_at = datetime.now(timezone.utc).isoformat()
        template.provenance_hash = _compute_hash(template)

        # Record metrics
        record_template(template.framework, "updated")
        record_processing_duration("update_template", time.time() - start_time)

        # Record provenance
        self.provenance.record(
            entity_type="template",
            entity_id=template_id,
            action="update",
            data_hash=template.provenance_hash,
        )

        # Update statistics
        self._stats.active_templates = sum(
            1 for t in self._templates.values() if t.status in ("draft", "active")
        )

        logger.info("Updated template %s", template_id)
        return template

    def clone_template(
        self,
        template_id: str,
        new_name: Optional[str] = None,
        new_version: Optional[str] = None,
    ) -> QuestionnaireTemplate:
        """Clone an existing template to create a new version.

        Args:
            template_id: Source template identifier.
            new_name: Optional new name for cloned template.
            new_version: Optional new version string.

        Returns:
            New QuestionnaireTemplate (clone).

        Raises:
            ValueError: If source template not found.
        """
        source = self._templates.get(template_id)
        if source is None:
            raise ValueError(f"Template {template_id} not found")

        cloned = self.create_template(
            name=new_name or f"{source.name} (Clone)",
            framework=source.framework,
            version=new_version or source.version,
            description=source.description,
            sections=source.sections,
            language=source.language,
            tags=source.tags,
        )

        record_template(source.framework, "cloned")

        logger.info(
            "Cloned template %s -> %s", template_id, cloned.template_id,
        )
        return cloned

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    def distribute(
        self,
        template_id: str,
        supplier_id: str,
        supplier_name: str,
        supplier_email: str,
        campaign_id: Optional[str] = None,
        channel: str = "email",
        deadline: Optional[str] = None,
    ) -> Distribution:
        """Distribute a questionnaire to a supplier.

        Args:
            template_id: Template identifier to distribute.
            supplier_id: Target supplier identifier.
            supplier_name: Target supplier display name.
            supplier_email: Supplier contact email.
            campaign_id: Optional campaign identifier.
            channel: Distribution channel (email, portal, api, bulk).
            deadline: Response deadline (ISO 8601). Defaults to config default.

        Returns:
            Distribution with distribution details.

        Raises:
            ValueError: If template not found or supplier_id is empty.
        """
        start_time = time.time()

        template = self._templates.get(template_id)
        if template is None:
            raise ValueError(f"Template {template_id} not found")
        if not supplier_id.strip():
            raise ValueError("supplier_id must not be empty")

        # Generate campaign ID if not provided
        cid = campaign_id or str(uuid.uuid4())

        # Ensure campaign exists
        if cid not in self._campaigns:
            self._campaigns[cid] = {
                "campaign_id": cid,
                "template_id": template_id,
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._stats.total_campaigns += 1
            self._stats.active_campaigns += 1
            update_active_campaigns(1)

        # Default deadline from config
        if not deadline:
            from datetime import timedelta
            dl = datetime.now(timezone.utc) + timedelta(
                days=self.config.default_deadline_days,
            )
            deadline = dl.isoformat()

        dist = Distribution(
            template_id=template_id,
            campaign_id=cid,
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            supplier_email=supplier_email,
            channel=channel,
            status="sent",
            deadline=deadline,
        )
        dist.provenance_hash = _compute_hash(dist)

        self._distributions[dist.distribution_id] = dist

        # Record metrics
        record_distribution(channel, "sent")
        update_pending_responses(1)
        record_processing_duration("distribute", time.time() - start_time)

        # Record provenance
        self.provenance.record(
            entity_type="distribution",
            entity_id=dist.distribution_id,
            action="distribute",
            data_hash=dist.provenance_hash,
        )

        # Update statistics
        self._stats.total_distributions += 1

        logger.info(
            "Distributed questionnaire %s to supplier %s (%s) via %s",
            template_id, supplier_id, supplier_name, channel,
        )
        return dist

    def get_distribution(
        self,
        distribution_id: str,
    ) -> Optional[Distribution]:
        """Get a distribution by ID.

        Args:
            distribution_id: Distribution identifier.

        Returns:
            Distribution or None if not found.
        """
        return self._distributions.get(distribution_id)

    def list_distributions(
        self,
        campaign_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Distribution]:
        """List distributions with optional filters.

        Args:
            campaign_id: Optional campaign filter.
            supplier_id: Optional supplier filter.
            status: Optional status filter.
            limit: Maximum number of distributions to return.
            offset: Number of distributions to skip.

        Returns:
            List of Distribution instances.
        """
        dists = list(self._distributions.values())

        if campaign_id is not None:
            dists = [d for d in dists if d.campaign_id == campaign_id]
        if supplier_id is not None:
            dists = [d for d in dists if d.supplier_id == supplier_id]
        if status is not None:
            dists = [d for d in dists if d.status == status]

        return dists[offset:offset + limit]

    def create_campaign(
        self,
        template_id: str,
        name: str = "",
        description: str = "",
    ) -> Dict[str, Any]:
        """Create a new questionnaire distribution campaign.

        Args:
            template_id: Template identifier for this campaign.
            name: Campaign display name.
            description: Campaign description.

        Returns:
            Campaign record dict.

        Raises:
            ValueError: If template not found.
        """
        template = self._templates.get(template_id)
        if template is None:
            raise ValueError(f"Template {template_id} not found")

        campaign_id = str(uuid.uuid4())
        campaign = {
            "campaign_id": campaign_id,
            "template_id": template_id,
            "name": name,
            "description": description,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._campaigns[campaign_id] = campaign

        self._stats.total_campaigns += 1
        self._stats.active_campaigns += 1
        update_active_campaigns(1)

        self.provenance.record(
            entity_type="campaign",
            entity_id=campaign_id,
            action="create",
            data_hash=_compute_hash(campaign),
        )

        logger.info("Created campaign %s for template %s", campaign_id, template_id)
        return campaign

    # ------------------------------------------------------------------
    # Response management
    # ------------------------------------------------------------------

    def submit_response(
        self,
        distribution_id: str,
        supplier_id: str,
        supplier_name: str,
        answers: Dict[str, Any],
        evidence_files: Optional[List[str]] = None,
        channel: str = "portal",
    ) -> QuestionnaireResponse:
        """Submit a questionnaire response.

        Args:
            distribution_id: Linked distribution identifier.
            supplier_id: Responding supplier identifier.
            supplier_name: Responding supplier display name.
            answers: Answer data keyed by question_id.
            evidence_files: Attached evidence file references.
            channel: Response submission channel.

        Returns:
            QuestionnaireResponse with submission details.

        Raises:
            ValueError: If distribution not found.
        """
        start_time = time.time()

        dist = self._distributions.get(distribution_id)
        if dist is None:
            raise ValueError(f"Distribution {distribution_id} not found")

        # Calculate completion percentage
        template = self._templates.get(dist.template_id)
        total_questions = template.questions if template else 1
        completion_pct = (
            (len(answers) / max(total_questions, 1)) * 100.0
        )

        response = QuestionnaireResponse(
            distribution_id=distribution_id,
            template_id=dist.template_id,
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            answers=answers,
            completion_pct=min(completion_pct, 100.0),
            status="submitted",
            channel=channel,
            evidence_files=evidence_files or [],
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
        response.provenance_hash = _compute_hash(response)

        self._responses[response.response_id] = response

        # Record metrics
        record_response(channel, "submitted")
        update_pending_responses(-1)
        record_processing_duration("submit_response", time.time() - start_time)

        # Record provenance
        self.provenance.record(
            entity_type="response",
            entity_id=response.response_id,
            action="submit",
            data_hash=response.provenance_hash,
        )

        # Update statistics
        self._stats.total_responses += 1

        logger.info(
            "Response %s submitted by supplier %s (%.1f%% complete)",
            response.response_id, supplier_id, completion_pct,
        )
        return response

    def get_response(
        self,
        response_id: str,
    ) -> Optional[QuestionnaireResponse]:
        """Get a response by ID.

        Args:
            response_id: Response identifier.

        Returns:
            QuestionnaireResponse or None if not found.
        """
        return self._responses.get(response_id)

    def list_responses(
        self,
        supplier_id: Optional[str] = None,
        template_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[QuestionnaireResponse]:
        """List responses with optional filters.

        Args:
            supplier_id: Optional supplier filter.
            template_id: Optional template filter.
            status: Optional status filter.
            limit: Maximum number of responses to return.
            offset: Number of responses to skip.

        Returns:
            List of QuestionnaireResponse instances.
        """
        responses = list(self._responses.values())

        if supplier_id is not None:
            responses = [r for r in responses if r.supplier_id == supplier_id]
        if template_id is not None:
            responses = [r for r in responses if r.template_id == template_id]
        if status is not None:
            responses = [r for r in responses if r.status == status]

        return responses[offset:offset + limit]

    def update_response(
        self,
        response_id: str,
        answers: Optional[Dict[str, Any]] = None,
        evidence_files: Optional[List[str]] = None,
    ) -> QuestionnaireResponse:
        """Update an existing response (append/modify answers).

        Args:
            response_id: Response identifier.
            answers: Updated answers to merge.
            evidence_files: Updated evidence file references.

        Returns:
            Updated QuestionnaireResponse.

        Raises:
            ValueError: If response not found or already finalized.
        """
        response = self._responses.get(response_id)
        if response is None:
            raise ValueError(f"Response {response_id} not found")
        if response.status == "finalized":
            raise ValueError(f"Response {response_id} is already finalized")

        if answers is not None:
            response.answers.update(answers)
            # Recalculate completion
            template = self._templates.get(response.template_id)
            total_questions = template.questions if template else 1
            response.completion_pct = min(
                (len(response.answers) / max(total_questions, 1)) * 100.0,
                100.0,
            )

        if evidence_files is not None:
            response.evidence_files = evidence_files

        response.provenance_hash = _compute_hash(response)

        self.provenance.record(
            entity_type="response",
            entity_id=response_id,
            action="update",
            data_hash=response.provenance_hash,
        )

        logger.info("Updated response %s", response_id)
        return response

    def finalize_response(
        self,
        response_id: str,
    ) -> QuestionnaireResponse:
        """Finalize a response (lock it for scoring).

        Args:
            response_id: Response identifier.

        Returns:
            Finalized QuestionnaireResponse.

        Raises:
            ValueError: If response not found or already finalized.
        """
        response = self._responses.get(response_id)
        if response is None:
            raise ValueError(f"Response {response_id} not found")
        if response.status == "finalized":
            raise ValueError(f"Response {response_id} is already finalized")

        response.status = "finalized"
        response.finalized_at = datetime.now(timezone.utc).isoformat()
        response.provenance_hash = _compute_hash(response)

        record_response(response.channel, "finalized")

        self.provenance.record(
            entity_type="response",
            entity_id=response_id,
            action="finalize",
            data_hash=response.provenance_hash,
        )

        self._stats.total_finalized += 1

        logger.info("Finalized response %s", response_id)
        return response

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_response(
        self,
        response_id: str,
        level: str = "completeness",
    ) -> ValidationResult:
        """Validate a questionnaire response.

        Performs deterministic validation checks at the specified level.
        No LLM is used for validation logic (zero-hallucination).

        Args:
            response_id: Response identifier.
            level: Validation level (completeness, consistency, evidence, cross_field).

        Returns:
            ValidationResult with check outcomes.

        Raises:
            ValueError: If response not found.
        """
        start_time = time.time()

        response = self._responses.get(response_id)
        if response is None:
            raise ValueError(f"Response {response_id} not found")

        errors: List[str] = []
        warnings: List[str] = []
        checks_passed = 0
        checks_failed = 0
        checks_warned = 0

        # Level 1: Completeness check
        if response.completion_pct < self.config.min_completion_pct:
            errors.append(
                f"Completion {response.completion_pct:.1f}% is below "
                f"minimum {self.config.min_completion_pct:.1f}%"
            )
            checks_failed += 1
        else:
            checks_passed += 1

        # Check for empty answers
        empty_count = sum(
            1 for v in response.answers.values()
            if v is None or v == "" or v == []
        )
        if empty_count > 0:
            warnings.append(f"{empty_count} answers are empty or null")
            checks_warned += 1
        else:
            checks_passed += 1

        # Level 2: Consistency check
        if level in ("consistency", "evidence", "cross_field"):
            template = self._templates.get(response.template_id)
            if template is not None:
                # Validate all required sections are addressed
                for section in template.sections:
                    section_name = section.get("name", "Unknown")
                    section_questions = section.get("questions", [])
                    answered = sum(
                        1 for q in section_questions
                        if q.get("id") in response.answers
                    )
                    if answered == 0 and len(section_questions) > 0:
                        errors.append(
                            f"Section '{section_name}' has no answers"
                        )
                        checks_failed += 1
                    else:
                        checks_passed += 1

        # Level 3: Evidence check
        if level in ("evidence", "cross_field"):
            if not response.evidence_files:
                warnings.append("No evidence files attached")
                checks_warned += 1
            else:
                checks_passed += 1

        is_valid = checks_failed == 0

        result = ValidationResult(
            response_id=response_id,
            is_valid=is_valid,
            completion_pct=response.completion_pct,
            errors=errors,
            warnings=warnings,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            checks_warned=checks_warned,
            level=level,
        )
        result.provenance_hash = _compute_hash(result)

        self._validations[result.validation_id] = result

        # Record metrics
        record_validation(level, "pass" if is_valid else "fail")
        record_processing_duration("validate", time.time() - start_time)

        # Record provenance
        self.provenance.record(
            entity_type="validation",
            entity_id=result.validation_id,
            action="validate",
            data_hash=result.provenance_hash,
        )

        # Update statistics
        self._stats.total_validations += 1

        logger.info(
            "Validated response %s: valid=%s (passed=%d, failed=%d, warned=%d)",
            response_id, is_valid, checks_passed, checks_failed, checks_warned,
        )
        return result

    def batch_validate(
        self,
        response_ids: List[str],
        level: str = "completeness",
    ) -> List[ValidationResult]:
        """Validate multiple responses in batch.

        Args:
            response_ids: List of response identifiers to validate.
            level: Validation level to apply.

        Returns:
            List of ValidationResult instances.
        """
        results = []
        for rid in response_ids:
            try:
                result = self.validate_response(rid, level=level)
                results.append(result)
            except ValueError as exc:
                logger.warning("Batch validation skipped %s: %s", rid, exc)
        return results

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_response(
        self,
        response_id: str,
        framework: Optional[str] = None,
    ) -> ScoringResult:
        """Score a questionnaire response.

        Uses deterministic scoring formulas based on the framework.
        No LLM is used for score computation (zero-hallucination).

        Args:
            response_id: Response identifier.
            framework: Scoring framework override (uses template framework if None).

        Returns:
            ScoringResult with score details.

        Raises:
            ValueError: If response not found.
        """
        start_time = time.time()

        response = self._responses.get(response_id)
        if response is None:
            raise ValueError(f"Response {response_id} not found")

        template = self._templates.get(response.template_id)
        scoring_framework = framework or (
            template.framework if template else self.config.default_framework
        )

        # Deterministic scoring: calculate per-section and overall scores
        section_scores: Dict[str, float] = {}
        category_scores: Dict[str, float] = {}

        if template is not None:
            for section in template.sections:
                section_name = section.get("name", "Unknown")
                section_questions = section.get("questions", [])
                if not section_questions:
                    continue

                answered = sum(
                    1 for q in section_questions
                    if q.get("id") in response.answers
                    and response.answers[q["id"]] not in (None, "", [])
                )
                section_score = (answered / max(len(section_questions), 1)) * 100.0
                section_scores[section_name] = round(section_score, 2)

                # Category aggregation
                category = section.get("category", "general")
                if category not in category_scores:
                    category_scores[category] = 0.0
                category_scores[category] = max(
                    category_scores[category], section_score,
                )

        # Calculate total score as weighted average of section scores
        if section_scores:
            total_score = round(
                sum(section_scores.values()) / len(section_scores), 2,
            )
        else:
            total_score = round(response.completion_pct, 2)

        # Determine performance tier
        tier = self._determine_tier(total_score)

        # Calculate benchmark percentile (stub: based on all scored responses)
        all_scores = [s.total_score for s in self._scores.values()]
        if all_scores:
            below_count = sum(1 for s in all_scores if s < total_score)
            benchmark_pct = round(
                (below_count / len(all_scores)) * 100.0, 2,
            )
        else:
            benchmark_pct = 50.0

        result = ScoringResult(
            response_id=response_id,
            supplier_id=response.supplier_id,
            framework=scoring_framework,
            total_score=total_score,
            tier=tier,
            section_scores=section_scores,
            category_scores=category_scores,
            benchmark_percentile=benchmark_pct,
        )
        result.provenance_hash = _compute_hash(result)

        self._scores[result.score_id] = result

        # Record metrics
        record_score(scoring_framework, tier)
        record_data_quality(scoring_framework, total_score)
        record_processing_duration("score", time.time() - start_time)

        # Record provenance
        self.provenance.record(
            entity_type="score",
            entity_id=result.score_id,
            action="score",
            data_hash=result.provenance_hash,
        )

        # Update statistics
        self._stats.total_scores += 1
        self._update_avg_score(total_score)

        logger.info(
            "Scored response %s: %.2f (%s tier, %s framework)",
            response_id, total_score, tier, scoring_framework,
        )
        return result

    def get_score(self, score_id: str) -> Optional[ScoringResult]:
        """Get a scoring result by ID.

        Args:
            score_id: Score identifier.

        Returns:
            ScoringResult or None if not found.
        """
        return self._scores.get(score_id)

    def get_supplier_scores(
        self,
        supplier_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ScoringResult]:
        """Get all scores for a specific supplier.

        Args:
            supplier_id: Supplier identifier.
            limit: Maximum number of scores to return.
            offset: Number of scores to skip.

        Returns:
            List of ScoringResult instances.
        """
        scores = [
            s for s in self._scores.values()
            if s.supplier_id == supplier_id
        ]
        return scores[offset:offset + limit]

    def benchmark_supplier(
        self,
        supplier_id: str,
    ) -> Dict[str, Any]:
        """Benchmark a supplier against peers.

        Args:
            supplier_id: Supplier identifier to benchmark.

        Returns:
            Benchmark comparison dict.
        """
        supplier_scores = self.get_supplier_scores(supplier_id)
        all_scores = list(self._scores.values())

        if not supplier_scores:
            return {
                "supplier_id": supplier_id,
                "avg_score": 0.0,
                "peer_avg_score": 0.0,
                "percentile": 0.0,
                "tier": "lagging",
                "total_assessments": 0,
            }

        avg_score = sum(s.total_score for s in supplier_scores) / len(supplier_scores)
        peer_avg = (
            sum(s.total_score for s in all_scores) / len(all_scores)
            if all_scores else 0.0
        )
        below = sum(1 for s in all_scores if s.total_score < avg_score)
        percentile = (below / max(len(all_scores), 1)) * 100.0

        return {
            "supplier_id": supplier_id,
            "avg_score": round(avg_score, 2),
            "peer_avg_score": round(peer_avg, 2),
            "percentile": round(percentile, 2),
            "tier": self._determine_tier(avg_score),
            "total_assessments": len(supplier_scores),
        }

    # ------------------------------------------------------------------
    # Follow-up management
    # ------------------------------------------------------------------

    def schedule_reminders(
        self,
        campaign_id: str,
    ) -> List[FollowUpAction]:
        """Schedule reminders for all pending distributions in a campaign.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            List of scheduled FollowUpAction instances.
        """
        dists = [
            d for d in self._distributions.values()
            if d.campaign_id == campaign_id and d.status == "sent"
        ]

        # Filter to distributions without submitted responses
        responded_suppliers = {
            r.supplier_id for r in self._responses.values()
            if r.distribution_id in {d.distribution_id for d in dists}
        }

        actions: List[FollowUpAction] = []
        for dist in dists:
            if dist.supplier_id in responded_suppliers:
                continue
            if dist.reminder_count >= self.config.max_reminders:
                continue

            action = FollowUpAction(
                distribution_id=dist.distribution_id,
                campaign_id=campaign_id,
                supplier_id=dist.supplier_id,
                action_type="reminder",
                status="scheduled",
                message=f"Reminder: Please complete your questionnaire by {dist.deadline}",
            )
            action.provenance_hash = _compute_hash(action)
            self._followups[action.action_id] = action

            dist.reminder_count += 1
            record_followup("reminder", "scheduled")

            self.provenance.record(
                entity_type="followup",
                entity_id=action.action_id,
                action="schedule",
                data_hash=action.provenance_hash,
            )

            actions.append(action)
            self._stats.total_followups += 1

        logger.info(
            "Scheduled %d reminders for campaign %s", len(actions), campaign_id,
        )
        return actions

    def trigger_reminder(
        self,
        distribution_id: str,
        message: str = "",
    ) -> FollowUpAction:
        """Trigger a single reminder for a distribution.

        Args:
            distribution_id: Distribution identifier.
            message: Reminder message content.

        Returns:
            FollowUpAction with reminder details.

        Raises:
            ValueError: If distribution not found.
        """
        dist = self._distributions.get(distribution_id)
        if dist is None:
            raise ValueError(f"Distribution {distribution_id} not found")

        action = FollowUpAction(
            distribution_id=distribution_id,
            campaign_id=dist.campaign_id,
            supplier_id=dist.supplier_id,
            action_type="reminder",
            status="sent",
            executed_at=datetime.now(timezone.utc).isoformat(),
            message=message or f"Reminder: Please complete your questionnaire by {dist.deadline}",
        )
        action.provenance_hash = _compute_hash(action)
        self._followups[action.action_id] = action

        dist.reminder_count += 1
        record_followup("reminder", "sent")

        self.provenance.record(
            entity_type="followup",
            entity_id=action.action_id,
            action="trigger",
            data_hash=action.provenance_hash,
        )

        self._stats.total_followups += 1

        logger.info("Triggered reminder for distribution %s", distribution_id)
        return action

    def escalate(
        self,
        distribution_id: str,
        message: str = "",
    ) -> FollowUpAction:
        """Escalate a non-responsive distribution.

        Args:
            distribution_id: Distribution identifier.
            message: Escalation message content.

        Returns:
            FollowUpAction with escalation details.

        Raises:
            ValueError: If distribution not found.
        """
        dist = self._distributions.get(distribution_id)
        if dist is None:
            raise ValueError(f"Distribution {distribution_id} not found")

        action = FollowUpAction(
            distribution_id=distribution_id,
            campaign_id=dist.campaign_id,
            supplier_id=dist.supplier_id,
            action_type="escalation",
            status="sent",
            executed_at=datetime.now(timezone.utc).isoformat(),
            message=message or "Escalation: Questionnaire response is overdue",
        )
        action.provenance_hash = _compute_hash(action)
        self._followups[action.action_id] = action

        record_followup("escalation", "sent")

        self.provenance.record(
            entity_type="followup",
            entity_id=action.action_id,
            action="escalate",
            data_hash=action.provenance_hash,
        )

        self._stats.total_followups += 1

        logger.info("Escalated distribution %s", distribution_id)
        return action

    def get_due_reminders(
        self,
        campaign_id: str,
    ) -> List[FollowUpAction]:
        """Get scheduled but not yet sent reminders for a campaign.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            List of scheduled FollowUpAction instances.
        """
        return [
            f for f in self._followups.values()
            if f.campaign_id == campaign_id and f.status == "scheduled"
        ]

    # ------------------------------------------------------------------
    # Analytics and reporting
    # ------------------------------------------------------------------

    def get_campaign_analytics(
        self,
        campaign_id: str,
    ) -> CampaignAnalytics:
        """Get analytics for a questionnaire campaign.

        All analytics are deterministic aggregations of recorded data.
        No LLM is used for analytics computation (zero-hallucination).

        Args:
            campaign_id: Campaign identifier.

        Returns:
            CampaignAnalytics with summary metrics.
        """
        start_time = time.time()

        # Get distributions for campaign
        dists = [
            d for d in self._distributions.values()
            if d.campaign_id == campaign_id
        ]

        # Get responses linked to these distributions
        dist_ids = {d.distribution_id for d in dists}
        responses = [
            r for r in self._responses.values()
            if r.distribution_id in dist_ids
        ]

        finalized = [r for r in responses if r.status == "finalized"]

        total_distributed = len(dists)
        total_responded = len(responses)
        total_finalized = len(finalized)

        response_rate = (
            (total_responded / max(total_distributed, 1)) * 100.0
        )

        avg_completion = (
            sum(r.completion_pct for r in responses) / max(len(responses), 1)
        )

        # Get scores for these responses
        response_ids = {r.response_id for r in responses}
        scores = [
            s for s in self._scores.values()
            if s.response_id in response_ids
        ]

        avg_score = (
            sum(s.total_score for s in scores) / max(len(scores), 1)
        )

        # Score distribution by tier
        score_distribution = {"leader": 0, "advanced": 0, "developing": 0, "lagging": 0}
        for s in scores:
            if s.tier in score_distribution:
                score_distribution[s.tier] += 1

        # Identify compliance gaps (sections with low average scores)
        compliance_gaps: List[Dict[str, Any]] = []
        section_totals: Dict[str, List[float]] = {}
        for s in scores:
            for section_name, section_score in s.section_scores.items():
                if section_name not in section_totals:
                    section_totals[section_name] = []
                section_totals[section_name].append(section_score)

        for section_name, section_vals in section_totals.items():
            section_avg = sum(section_vals) / len(section_vals)
            if section_avg < 50.0:
                compliance_gaps.append({
                    "section": section_name,
                    "avg_score": round(section_avg, 2),
                    "supplier_count": len(section_vals),
                    "severity": "high" if section_avg < 25.0 else "medium",
                })

        analytics = CampaignAnalytics(
            campaign_id=campaign_id,
            total_distributed=total_distributed,
            total_responded=total_responded,
            total_finalized=total_finalized,
            response_rate_pct=round(response_rate, 2),
            avg_completion_pct=round(avg_completion, 2),
            avg_score=round(avg_score, 2),
            score_distribution=score_distribution,
            compliance_gaps=compliance_gaps,
        )
        analytics.provenance_hash = _compute_hash(analytics)

        # Update response rate metric
        update_response_rate(campaign_id, response_rate)

        record_processing_duration("analytics", time.time() - start_time)

        self.provenance.record(
            entity_type="analytics",
            entity_id=campaign_id,
            action="generate",
            data_hash=analytics.provenance_hash,
        )

        logger.info(
            "Generated analytics for campaign %s: %d distributed, "
            "%d responded (%.1f%%), avg score %.2f",
            campaign_id, total_distributed, total_responded,
            response_rate, avg_score,
        )
        return analytics

    def get_response_rate(
        self,
        campaign_id: str,
    ) -> float:
        """Get current response rate for a campaign.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            Response rate percentage (0.0 - 100.0).
        """
        dists = [
            d for d in self._distributions.values()
            if d.campaign_id == campaign_id
        ]
        dist_ids = {d.distribution_id for d in dists}
        responses = [
            r for r in self._responses.values()
            if r.distribution_id in dist_ids
        ]

        rate = (len(responses) / max(len(dists), 1)) * 100.0
        return round(rate, 2)

    def generate_report(
        self,
        campaign_id: str,
    ) -> Dict[str, Any]:
        """Generate a comprehensive report for a campaign.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            Report dict with analytics, scores, and gaps.
        """
        analytics = self.get_campaign_analytics(campaign_id)
        campaign = self._campaigns.get(campaign_id, {})

        return {
            "campaign": campaign,
            "analytics": analytics.model_dump(mode="json"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance_hash": _compute_hash(analytics),
        }

    def get_compliance_gaps(
        self,
        campaign_id: str,
    ) -> List[Dict[str, Any]]:
        """Get compliance gaps identified in a campaign.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            List of compliance gap records.
        """
        analytics = self.get_campaign_analytics(campaign_id)
        return analytics.compliance_gaps

    # ------------------------------------------------------------------
    # Statistics and health
    # ------------------------------------------------------------------

    def get_statistics(self) -> QuestionnaireStatistics:
        """Get aggregated supplier questionnaire statistics.

        Returns:
            QuestionnaireStatistics summary.
        """
        return self._stats

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the service.

        Returns:
            Health status dict.
        """
        return {
            "status": "healthy" if self._started else "not_started",
            "service": "supplier-questionnaire",
            "started": self._started,
            "templates": len(self._templates),
            "distributions": len(self._distributions),
            "responses": len(self._responses),
            "scores": len(self._scores),
            "provenance_entries": self.provenance.entry_count,
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    def get_provenance(self) -> _ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            _ProvenanceTracker used by this service.
        """
        return self.provenance

    def get_metrics(self) -> Dict[str, Any]:
        """Get supplier questionnaire service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_templates": self._stats.total_templates,
            "active_templates": self._stats.active_templates,
            "total_distributions": self._stats.total_distributions,
            "total_responses": self._stats.total_responses,
            "total_finalized": self._stats.total_finalized,
            "total_validations": self._stats.total_validations,
            "total_scores": self._stats.total_scores,
            "total_followups": self._stats.total_followups,
            "total_campaigns": self._stats.total_campaigns,
            "active_campaigns": self._stats.active_campaigns,
            "provenance_entries": self.provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_tier(self, score: float) -> str:
        """Determine performance tier from a score.

        Uses deterministic thresholds from configuration.

        Args:
            score: Score value (0-100).

        Returns:
            Performance tier string (leader, advanced, developing, lagging).
        """
        if score >= self.config.score_leader_threshold:
            return "leader"
        elif score >= self.config.score_advanced_threshold:
            return "advanced"
        elif score >= self.config.score_developing_threshold:
            return "developing"
        else:
            return "lagging"

    def _update_avg_score(self, score: float) -> None:
        """Update running average score.

        Args:
            score: Latest score value.
        """
        total = self._stats.total_scores
        if total <= 0:
            self._stats.avg_score = score
            return
        prev_avg = self._stats.avg_score
        self._stats.avg_score = (
            (prev_avg * (total - 1) + score) / total
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the supplier questionnaire service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("SupplierQuestionnaireService already started; skipping")
            return

        logger.info("SupplierQuestionnaireService starting up...")
        self._started = True
        logger.info("SupplierQuestionnaireService startup complete")

    def shutdown(self) -> None:
        """Shutdown the supplier questionnaire service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("SupplierQuestionnaireService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> SupplierQuestionnaireService:
    """Get or create the singleton SupplierQuestionnaireService instance.

    Returns:
        The singleton SupplierQuestionnaireService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = SupplierQuestionnaireService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_supplier_questionnaire(
    app: Any,
    config: Optional[SupplierQuestionnaireConfig] = None,
) -> SupplierQuestionnaireService:
    """Configure the Supplier Questionnaire Service on a FastAPI application.

    Creates the SupplierQuestionnaireService, stores it in app.state, mounts
    the supplier questionnaire API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional supplier questionnaire config.

    Returns:
        SupplierQuestionnaireService instance.
    """
    global _singleton_instance

    service = SupplierQuestionnaireService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.supplier_questionnaire_service = service

    # Mount supplier questionnaire API router
    try:
        from greenlang.supplier_questionnaire.api.router import router as quest_router
        if quest_router is not None:
            app.include_router(quest_router)
            logger.info("Supplier questionnaire service API router mounted")
    except ImportError:
        logger.warning(
            "Supplier questionnaire router not available; API not mounted"
        )

    # Start service
    service.startup()

    logger.info("Supplier questionnaire service configured on app")
    return service


def get_supplier_questionnaire(app: Any) -> SupplierQuestionnaireService:
    """Get the SupplierQuestionnaireService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        SupplierQuestionnaireService instance.

    Raises:
        RuntimeError: If supplier questionnaire service not configured.
    """
    service = getattr(app.state, "supplier_questionnaire_service", None)
    if service is None:
        raise RuntimeError(
            "Supplier questionnaire service not configured. "
            "Call configure_supplier_questionnaire(app) first."
        )
    return service


def get_router(service: Optional[SupplierQuestionnaireService] = None) -> Any:
    """Get the supplier questionnaire API router.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.supplier_questionnaire.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "SupplierQuestionnaireService",
    "configure_supplier_questionnaire",
    "get_supplier_questionnaire",
    "get_router",
    # Models
    "QuestionnaireTemplate",
    "Distribution",
    "QuestionnaireResponse",
    "ValidationResult",
    "ScoringResult",
    "FollowUpAction",
    "CampaignAnalytics",
    "QuestionnaireStatistics",
]
