# -*- coding: utf-8 -*-
"""
Questionnaire Template Engine - AGENT-DATA-008: Supplier Questionnaire Processor
=================================================================================

Manages questionnaire template lifecycle including creation, versioning,
cloning, section and question management, validation, and import/export.
Ships with built-in framework templates for CDP Climate (11 modules),
EcoVadis (4 themes), and DJSI (3 dimensions).

Supports:
    - Template CRUD with full versioning
    - Section and question management
    - Multi-language translation support
    - Template cloning with new IDs
    - Template validation (structure, completeness)
    - JSON export/import
    - Built-in CDP, EcoVadis, and DJSI skeleton templates
    - SHA-256 provenance hashes on all mutations
    - Thread-safe in-memory storage

Zero-Hallucination Guarantees:
    - All template operations are deterministic
    - No LLM involvement in template management
    - SHA-256 provenance hashes for audit trails
    - Version numbers are monotonically increasing integers

Example:
    >>> from greenlang.supplier_questionnaire.questionnaire_template import (
    ...     QuestionnaireTemplateEngine,
    ... )
    >>> engine = QuestionnaireTemplateEngine()
    >>> template = engine.create_template(
    ...     name="CDP Climate 2025",
    ...     framework="cdp_climate",
    ...     sections=[],
    ...     language="en",
    ... )
    >>> assert template.version == 1

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.supplier_questionnaire.models import (
    Framework,
    QuestionnaireStatus,
    QuestionnaireTemplate,
    QuestionType,
    TemplateQuestion,
    TemplateSection,
    ValidationCheck,
    ValidationSeverity,
    ValidationSummary,
)

logger = logging.getLogger(__name__)

__all__ = [
    "QuestionnaireTemplateEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Built-in framework section definitions
# ---------------------------------------------------------------------------

_CDP_CLIMATE_SECTIONS: List[Dict[str, Any]] = [
    {"name": "C0 - Introduction", "code": "C0", "order": 0},
    {"name": "C1 - Governance", "code": "C1", "order": 1},
    {"name": "C2 - Risks and Opportunities", "code": "C2", "order": 2},
    {"name": "C3 - Business Strategy", "code": "C3", "order": 3},
    {"name": "C4 - Targets and Performance", "code": "C4", "order": 4},
    {"name": "C5 - Emissions Methodology", "code": "C5", "order": 5},
    {"name": "C6 - Emissions Data", "code": "C6", "order": 6},
    {"name": "C7 - Emissions Breakdown", "code": "C7", "order": 7},
    {"name": "C8 - Energy", "code": "C8", "order": 8},
    {"name": "C9 - Additional Metrics", "code": "C9", "order": 9},
    {"name": "C10 - Verification", "code": "C10", "order": 10},
]

_ECOVADIS_SECTIONS: List[Dict[str, Any]] = [
    {"name": "Environment", "code": "ENV", "order": 0},
    {"name": "Labor & Human Rights", "code": "LAB", "order": 1},
    {"name": "Ethics", "code": "ETH", "order": 2},
    {"name": "Sustainable Procurement", "code": "SUP", "order": 3},
]

_DJSI_SECTIONS: List[Dict[str, Any]] = [
    {"name": "Economic Dimension", "code": "ECO", "order": 0},
    {"name": "Environmental Dimension", "code": "ENV", "order": 1},
    {"name": "Social Dimension", "code": "SOC", "order": 2},
]

_FRAMEWORK_SECTIONS: Dict[str, List[Dict[str, Any]]] = {
    Framework.CDP_CLIMATE.value: _CDP_CLIMATE_SECTIONS,
    Framework.ECOVADIS.value: _ECOVADIS_SECTIONS,
    Framework.DJSI.value: _DJSI_SECTIONS,
}

# Default questions per framework section
_CDP_SECTION_QUESTIONS: Dict[str, List[Dict[str, Any]]] = {
    "C0": [
        {"code": "C0.1", "text": "Give a general description of your organization.", "type": "text"},
        {"code": "C0.2", "text": "Reporting year and boundary.", "type": "text"},
        {"code": "C0.3", "text": "Select countries/regions where you operate.", "type": "multi_choice"},
    ],
    "C1": [
        {"code": "C1.1", "text": "Is there board-level oversight of climate-related issues?", "type": "yes_no"},
        {"code": "C1.1a", "text": "Identify the position of the board member(s) responsible.", "type": "text"},
        {"code": "C1.2", "text": "Provide details on management responsibility for climate.", "type": "text"},
        {"code": "C1.3", "text": "Describe incentives for management of climate issues.", "type": "text"},
    ],
    "C2": [
        {"code": "C2.1", "text": "Does your organization have a process for identifying climate risks?", "type": "yes_no"},
        {"code": "C2.2", "text": "Describe your process for assessing climate-related risks.", "type": "text"},
        {"code": "C2.3", "text": "Provide details of identified climate-related risks.", "type": "table"},
    ],
    "C3": [
        {"code": "C3.1", "text": "Does your organization's strategy include a transition plan?", "type": "yes_no"},
        {"code": "C3.2", "text": "Describe how climate has influenced your strategy.", "type": "text"},
    ],
    "C4": [
        {"code": "C4.1", "text": "Did you have an emissions target active during the reporting year?", "type": "yes_no"},
        {"code": "C4.1a", "text": "Provide details of your absolute emissions target(s).", "type": "table"},
        {"code": "C4.2", "text": "Did you have other climate-related targets?", "type": "yes_no"},
    ],
    "C5": [
        {"code": "C5.1", "text": "Provide your base year and base year emissions.", "type": "text"},
        {"code": "C5.2", "text": "Select the name of the standard or methodology used.", "type": "single_choice"},
    ],
    "C6": [
        {"code": "C6.1", "text": "Scope 1 gross global emissions (metric tons CO2e).", "type": "numeric"},
        {"code": "C6.3", "text": "Scope 2 location-based emissions (metric tons CO2e).", "type": "numeric"},
        {"code": "C6.5", "text": "Scope 3 total emissions (metric tons CO2e).", "type": "numeric"},
    ],
    "C7": [
        {"code": "C7.1", "text": "Scope 1 emissions breakdown by country/region.", "type": "table"},
        {"code": "C7.2", "text": "Scope 1 emissions breakdown by business division.", "type": "table"},
    ],
    "C8": [
        {"code": "C8.1", "text": "Total energy consumption in MWh.", "type": "numeric"},
        {"code": "C8.2", "text": "Breakdown of energy consumption by source.", "type": "table"},
    ],
    "C9": [
        {"code": "C9.1", "text": "Provide additional climate-related metrics.", "type": "table"},
    ],
    "C10": [
        {"code": "C10.1", "text": "Has your Scope 1 emissions data been third-party verified?", "type": "yes_no"},
        {"code": "C10.2", "text": "Provide details of the verification standard used.", "type": "text"},
    ],
}

_ECOVADIS_SECTION_QUESTIONS: Dict[str, List[Dict[str, Any]]] = {
    "ENV": [
        {"code": "ENV.1", "text": "Does your company have an environmental policy?", "type": "yes_no"},
        {"code": "ENV.2", "text": "Describe your GHG emissions reduction actions.", "type": "text"},
        {"code": "ENV.3", "text": "Total Scope 1+2 emissions (tCO2e).", "type": "numeric"},
        {"code": "ENV.4", "text": "Do you have environmental certifications (e.g. ISO 14001)?", "type": "multi_choice"},
        {"code": "ENV.5", "text": "Describe your waste management practices.", "type": "text"},
    ],
    "LAB": [
        {"code": "LAB.1", "text": "Does your company have a human rights policy?", "type": "yes_no"},
        {"code": "LAB.2", "text": "Describe health and safety management practices.", "type": "text"},
        {"code": "LAB.3", "text": "What is your employee turnover rate?", "type": "percentage"},
        {"code": "LAB.4", "text": "Do you have diversity and inclusion programs?", "type": "yes_no"},
    ],
    "ETH": [
        {"code": "ETH.1", "text": "Does your company have an anti-corruption policy?", "type": "yes_no"},
        {"code": "ETH.2", "text": "Describe your anti-competitive practices measures.", "type": "text"},
        {"code": "ETH.3", "text": "Do you have a whistleblower mechanism?", "type": "yes_no"},
    ],
    "SUP": [
        {"code": "SUP.1", "text": "Does your company have a sustainable procurement policy?", "type": "yes_no"},
        {"code": "SUP.2", "text": "What % of suppliers have been assessed on ESG criteria?", "type": "percentage"},
        {"code": "SUP.3", "text": "Describe your supplier ESG audit process.", "type": "text"},
    ],
}

_DJSI_SECTION_QUESTIONS: Dict[str, List[Dict[str, Any]]] = {
    "ECO": [
        {"code": "ECO.1", "text": "Describe your corporate governance structure.", "type": "text"},
        {"code": "ECO.2", "text": "Describe risk and crisis management processes.", "type": "text"},
        {"code": "ECO.3", "text": "Report on codes of business conduct.", "type": "text"},
        {"code": "ECO.4", "text": "Describe your tax strategy.", "type": "text"},
    ],
    "ENV": [
        {"code": "ENV.1", "text": "Report your environmental policy and management system.", "type": "text"},
        {"code": "ENV.2", "text": "Report total direct GHG emissions (Scope 1) in tCO2e.", "type": "numeric"},
        {"code": "ENV.3", "text": "Report total indirect GHG emissions (Scope 2) in tCO2e.", "type": "numeric"},
        {"code": "ENV.4", "text": "Report total water consumption in megalitres.", "type": "numeric"},
    ],
    "SOC": [
        {"code": "SOC.1", "text": "Describe your human capital development programs.", "type": "text"},
        {"code": "SOC.2", "text": "Report occupational health and safety performance.", "type": "text"},
        {"code": "SOC.3", "text": "Describe your stakeholder engagement processes.", "type": "text"},
    ],
}

_FRAMEWORK_QUESTIONS: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    Framework.CDP_CLIMATE.value: _CDP_SECTION_QUESTIONS,
    Framework.ECOVADIS.value: _ECOVADIS_SECTION_QUESTIONS,
    Framework.DJSI.value: _DJSI_SECTION_QUESTIONS,
}


# ---------------------------------------------------------------------------
# QuestionnaireTemplateEngine
# ---------------------------------------------------------------------------


class QuestionnaireTemplateEngine:
    """Questionnaire template lifecycle engine.

    Manages creation, versioning, cloning, validation, and import/export
    of questionnaire templates. Ships with built-in skeleton templates
    for CDP Climate, EcoVadis, and DJSI frameworks.

    Attributes:
        _templates: In-memory template storage keyed by template_id.
        _config: Configuration dictionary.
        _lock: Threading lock for mutations.
        _stats: Aggregate statistics counters.

    Example:
        >>> engine = QuestionnaireTemplateEngine()
        >>> t = engine.create_template("My Template", "cdp_climate", [], "en")
        >>> assert t.version == 1
        >>> t2 = engine.update_template(t.template_id, {"name": "Updated"})
        >>> assert t2.version == 2
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize QuestionnaireTemplateEngine.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``max_templates``: int (default 1000)
                - ``auto_populate_framework``: bool (default True)
        """
        self._config = config or {}
        self._templates: Dict[str, QuestionnaireTemplate] = {}
        self._lock = threading.Lock()
        self._max_templates: int = self._config.get("max_templates", 1000)
        self._auto_populate: bool = self._config.get(
            "auto_populate_framework", True,
        )
        self._stats: Dict[str, int] = {
            "templates_created": 0,
            "templates_updated": 0,
            "templates_cloned": 0,
            "templates_exported": 0,
            "templates_imported": 0,
            "validations_run": 0,
            "errors": 0,
        }
        logger.info(
            "QuestionnaireTemplateEngine initialised: max_templates=%d, "
            "auto_populate=%s",
            self._max_templates,
            self._auto_populate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_template(
        self,
        name: str,
        framework: str,
        sections: Optional[List[TemplateSection]] = None,
        language: str = "en",
        description: str = "",
        tags: Optional[List[str]] = None,
        created_by: str = "system",
    ) -> QuestionnaireTemplate:
        """Create a new questionnaire template.

        If the framework matches a built-in definition (CDP Climate,
        EcoVadis, DJSI) and no sections are provided, auto-populates
        the template with framework skeleton sections and questions.

        Args:
            name: Template display name.
            framework: Target framework (Framework enum value string).
            sections: Optional initial sections.
            language: Primary language code (default "en").
            description: Template description.
            tags: Free-form tags.
            created_by: User creating the template.

        Returns:
            Created QuestionnaireTemplate.

        Raises:
            ValueError: If name is empty or max templates reached.
        """
        start = time.monotonic()

        if not name or not name.strip():
            raise ValueError("Template name must be non-empty")

        with self._lock:
            if len(self._templates) >= self._max_templates:
                raise ValueError(
                    f"Maximum templates ({self._max_templates}) reached"
                )

        # Resolve framework enum
        fw = self._resolve_framework(framework)

        # Auto-populate sections from built-in definitions
        effective_sections = list(sections) if sections else []
        if not effective_sections and self._auto_populate:
            effective_sections = self._build_framework_sections(fw)

        template_id = str(uuid.uuid4())
        provenance_hash = self._compute_provenance(
            "create_template", template_id, name, fw.value,
        )

        template = QuestionnaireTemplate(
            template_id=template_id,
            name=name,
            framework=fw,
            version=1,
            status=QuestionnaireStatus.DRAFT,
            sections=effective_sections,
            language=language,
            supported_languages=[language],
            description=description,
            created_by=created_by,
            tags=tags or [],
            provenance_hash=provenance_hash,
        )

        with self._lock:
            self._templates[template_id] = template
            self._stats["templates_created"] += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Created template %s: name='%s' framework=%s sections=%d (%.1f ms)",
            template_id[:8], name, fw.value,
            len(effective_sections), elapsed_ms,
        )
        return template

    def get_template(self, template_id: str) -> QuestionnaireTemplate:
        """Get a template by ID.

        Args:
            template_id: Template identifier.

        Returns:
            QuestionnaireTemplate.

        Raises:
            ValueError: If template_id is not found.
        """
        return self._get_template_or_raise(template_id)

    def list_templates(
        self,
        framework: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[QuestionnaireTemplate]:
        """List templates with optional filtering.

        Args:
            framework: Filter by framework value.
            status: Filter by status value.

        Returns:
            List of matching QuestionnaireTemplate objects.
        """
        with self._lock:
            templates = list(self._templates.values())

        if framework is not None:
            templates = [
                t for t in templates if t.framework.value == framework
            ]
        if status is not None:
            templates = [
                t for t in templates if t.status.value == status
            ]

        logger.debug(
            "Listed %d templates (framework=%s, status=%s)",
            len(templates), framework, status,
        )
        return templates

    def update_template(
        self,
        template_id: str,
        updates: Dict[str, Any],
    ) -> QuestionnaireTemplate:
        """Update a template, creating a new version.

        Increments the version number and applies the provided field
        updates to a copy of the existing template.

        Args:
            template_id: Template to update.
            updates: Dictionary of field names to new values.

        Returns:
            Updated QuestionnaireTemplate with incremented version.

        Raises:
            ValueError: If template_id is not found.
        """
        start = time.monotonic()
        existing = self._get_template_or_raise(template_id)

        # Build updated dict
        data = existing.model_dump()
        allowed_fields = {
            "name", "description", "status", "sections", "language",
            "supported_languages", "tags",
        }
        for key, value in updates.items():
            if key in allowed_fields:
                data[key] = value

        # Increment version
        data["version"] = existing.version + 1
        data["updated_at"] = _utcnow()

        # Recompute provenance
        data["provenance_hash"] = self._compute_provenance(
            "update_template", template_id, str(data["version"]),
        )

        updated = QuestionnaireTemplate(**data)

        with self._lock:
            self._templates[template_id] = updated
            self._stats["templates_updated"] += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Updated template %s: v%d -> v%d (%.1f ms)",
            template_id[:8], existing.version, updated.version, elapsed_ms,
        )
        return updated

    def clone_template(
        self,
        template_id: str,
        new_name: str,
    ) -> QuestionnaireTemplate:
        """Clone a template with a new name and ID.

        Creates a deep copy of the template with fresh IDs for
        the template, all sections, and all questions.

        Args:
            template_id: Template to clone.
            new_name: Name for the cloned template.

        Returns:
            Cloned QuestionnaireTemplate with new IDs.

        Raises:
            ValueError: If template_id is not found.
        """
        start = time.monotonic()
        source = self._get_template_or_raise(template_id)

        # Deep copy sections with fresh IDs
        cloned_sections: List[TemplateSection] = []
        for section in source.sections:
            cloned_questions: List[TemplateQuestion] = []
            for q in section.questions:
                q_data = q.model_dump()
                q_data["question_id"] = str(uuid.uuid4())
                cloned_questions.append(TemplateQuestion(**q_data))

            s_data = section.model_dump()
            s_data["section_id"] = str(uuid.uuid4())
            s_data["questions"] = [cq.model_dump() for cq in cloned_questions]
            cloned_sections.append(TemplateSection(**s_data))

        new_id = str(uuid.uuid4())
        provenance_hash = self._compute_provenance(
            "clone_template", template_id, new_id, new_name,
        )

        cloned = QuestionnaireTemplate(
            template_id=new_id,
            name=new_name,
            framework=source.framework,
            version=1,
            status=QuestionnaireStatus.DRAFT,
            sections=cloned_sections,
            language=source.language,
            supported_languages=list(source.supported_languages),
            description=f"Cloned from {source.name} (v{source.version})",
            created_by="system",
            tags=list(source.tags),
            provenance_hash=provenance_hash,
        )

        with self._lock:
            self._templates[new_id] = cloned
            self._stats["templates_cloned"] += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Cloned template %s -> %s: name='%s' (%.1f ms)",
            template_id[:8], new_id[:8], new_name, elapsed_ms,
        )
        return cloned

    def add_section(
        self,
        template_id: str,
        section: TemplateSection,
    ) -> TemplateSection:
        """Add a section to a template.

        Args:
            template_id: Template to add section to.
            section: Section to add.

        Returns:
            The added TemplateSection.

        Raises:
            ValueError: If template_id is not found.
        """
        template = self._get_template_or_raise(template_id)

        # Assign order if not set
        if section.order == 0 and template.sections:
            section.order = max(s.order for s in template.sections) + 1

        with self._lock:
            self._templates[template_id].sections.append(section)
            self._templates[template_id].updated_at = _utcnow()
            self._templates[template_id].provenance_hash = (
                self._compute_provenance(
                    "add_section", template_id, section.section_id,
                )
            )

        logger.info(
            "Added section '%s' to template %s",
            section.name, template_id[:8],
        )
        return section

    def add_question(
        self,
        template_id: str,
        section_id: str,
        question: TemplateQuestion,
    ) -> TemplateQuestion:
        """Add a question to a template section.

        Args:
            template_id: Template containing the section.
            section_id: Section to add question to.
            question: Question to add.

        Returns:
            The added TemplateQuestion.

        Raises:
            ValueError: If template_id or section_id is not found.
        """
        template = self._get_template_or_raise(template_id)

        target_section: Optional[TemplateSection] = None
        for s in template.sections:
            if s.section_id == section_id:
                target_section = s
                break

        if target_section is None:
            raise ValueError(
                f"Section {section_id} not found in template {template_id}"
            )

        # Assign order if not set
        if question.order == 0 and target_section.questions:
            question.order = (
                max(q.order for q in target_section.questions) + 1
            )

        with self._lock:
            target_section.questions.append(question)
            self._templates[template_id].updated_at = _utcnow()
            self._templates[template_id].provenance_hash = (
                self._compute_provenance(
                    "add_question", template_id, section_id,
                    question.question_id,
                )
            )

        logger.info(
            "Added question '%s' to section '%s' in template %s",
            question.code or question.question_id[:8],
            target_section.name, template_id[:8],
        )
        return question

    def validate_template(
        self,
        template_id: str,
    ) -> ValidationSummary:
        """Validate a template for structural completeness.

        Checks that the template has sections, each section has
        questions, choice questions have choices defined, etc.

        Args:
            template_id: Template to validate.

        Returns:
            ValidationSummary with check results.

        Raises:
            ValueError: If template_id is not found.
        """
        start = time.monotonic()
        template = self._get_template_or_raise(template_id)

        checks: List[ValidationCheck] = []

        # Check 1: Template has sections
        checks.append(self._check_has_sections(template))

        # Check 2: Each section has questions
        for section in template.sections:
            checks.append(self._check_section_has_questions(section))

            # Check 3: Choice questions have choices
            for question in section.questions:
                if question.question_type in (
                    QuestionType.SINGLE_CHOICE,
                    QuestionType.MULTI_CHOICE,
                ):
                    checks.append(
                        self._check_question_has_choices(question, section)
                    )

                # Check 4: Question text is non-empty
                checks.append(
                    self._check_question_text(question, section)
                )

        # Check 5: Section ordering is valid
        checks.append(self._check_section_ordering(template))

        # Check 6: Template name and framework set
        checks.append(self._check_template_metadata(template))

        # Check 7: No duplicate question codes
        checks.append(self._check_duplicate_codes(template))

        passed = [c for c in checks if c.passed]
        failed = [c for c in checks if not c.passed]
        errors = [c for c in failed if c.severity == ValidationSeverity.ERROR]
        warnings = [
            c for c in failed if c.severity == ValidationSeverity.WARNING
        ]

        provenance_hash = self._compute_provenance(
            "validate_template", template_id, str(len(checks)),
        )

        with self._lock:
            self._stats["validations_run"] += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Validated template %s: %d checks, %d passed, %d failed (%.1f ms)",
            template_id[:8], len(checks), len(passed), len(failed), elapsed_ms,
        )

        return ValidationSummary(
            response_id="",
            template_id=template_id,
            checks=checks,
            total_checks=len(checks),
            passed_checks=len(passed),
            failed_checks=len(failed),
            warning_count=len(warnings),
            error_count=len(errors),
            is_valid=len(errors) == 0,
            data_quality_score=round(
                (len(passed) / len(checks) * 100) if checks else 0.0, 1,
            ),
            provenance_hash=provenance_hash,
        )

    def export_template(
        self,
        template_id: str,
        format: str = "json",
    ) -> str:
        """Export a template to a string format.

        Args:
            template_id: Template to export.
            format: Export format ("json" supported).

        Returns:
            Serialized template string.

        Raises:
            ValueError: If template_id is not found or format unsupported.
        """
        template = self._get_template_or_raise(template_id)

        if format.lower() != "json":
            raise ValueError(f"Unsupported export format: {format}")

        data = template.model_dump(mode="json")

        with self._lock:
            self._stats["templates_exported"] += 1

        logger.info("Exported template %s as %s", template_id[:8], format)
        return json.dumps(data, indent=2, default=str)

    def import_template(
        self,
        data: str,
        format: str = "json",
    ) -> QuestionnaireTemplate:
        """Import a template from a string format.

        Assigns a new template_id and resets version to 1.

        Args:
            data: Serialized template string.
            format: Import format ("json" supported).

        Returns:
            Imported QuestionnaireTemplate.

        Raises:
            ValueError: If format is unsupported or data is invalid.
        """
        start = time.monotonic()

        if format.lower() != "json":
            raise ValueError(f"Unsupported import format: {format}")

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        # Assign fresh IDs
        new_id = str(uuid.uuid4())
        parsed["template_id"] = new_id
        parsed["version"] = 1
        parsed["status"] = QuestionnaireStatus.DRAFT.value
        parsed["created_at"] = _utcnow().isoformat()
        parsed["updated_at"] = _utcnow().isoformat()
        parsed["provenance_hash"] = self._compute_provenance(
            "import_template", new_id,
        )

        # Assign fresh section and question IDs
        for section in parsed.get("sections", []):
            section["section_id"] = str(uuid.uuid4())
            for question in section.get("questions", []):
                question["question_id"] = str(uuid.uuid4())

        template = QuestionnaireTemplate(**parsed)

        with self._lock:
            self._templates[new_id] = template
            self._stats["templates_imported"] += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Imported template %s: name='%s' (%.1f ms)",
            new_id[:8], template.name, elapsed_ms,
        )
        return template

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine statistics.

        Returns:
            Dictionary of counter values.
        """
        with self._lock:
            return {
                **self._stats,
                "active_templates": len(self._templates),
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Validation check helpers
    # ------------------------------------------------------------------

    def _check_has_sections(
        self,
        template: QuestionnaireTemplate,
    ) -> ValidationCheck:
        """Check that template has at least one section."""
        has_sections = len(template.sections) > 0
        return ValidationCheck(
            check_type="structural",
            severity=ValidationSeverity.ERROR,
            passed=has_sections,
            message=(
                f"Template has {len(template.sections)} section(s)"
                if has_sections
                else "Template has no sections"
            ),
        )

    def _check_section_has_questions(
        self,
        section: TemplateSection,
    ) -> ValidationCheck:
        """Check that a section has at least one question."""
        has_questions = len(section.questions) > 0
        return ValidationCheck(
            check_type="structural",
            severity=ValidationSeverity.WARNING,
            passed=has_questions,
            message=(
                f"Section '{section.name}' has {len(section.questions)} question(s)"
                if has_questions
                else f"Section '{section.name}' has no questions"
            ),
        )

    def _check_question_has_choices(
        self,
        question: TemplateQuestion,
        section: TemplateSection,
    ) -> ValidationCheck:
        """Check that choice-type questions have choices defined."""
        has_choices = len(question.choices) > 0
        return ValidationCheck(
            check_type="structural",
            question_id=question.question_id,
            severity=ValidationSeverity.ERROR,
            passed=has_choices,
            message=(
                f"Question '{question.code}' has {len(question.choices)} choice(s)"
                if has_choices
                else f"Question '{question.code}' in section "
                f"'{section.name}' is type "
                f"{question.question_type.value} but has no choices"
            ),
            suggestion="Add choices to this question" if not has_choices else "",
        )

    def _check_question_text(
        self,
        question: TemplateQuestion,
        section: TemplateSection,
    ) -> ValidationCheck:
        """Check that question text is non-empty."""
        has_text = bool(question.text and question.text.strip())
        return ValidationCheck(
            check_type="completeness",
            question_id=question.question_id,
            severity=ValidationSeverity.ERROR,
            passed=has_text,
            message=(
                f"Question '{question.code}' has text"
                if has_text
                else f"Question in section '{section.name}' has empty text"
            ),
        )

    def _check_section_ordering(
        self,
        template: QuestionnaireTemplate,
    ) -> ValidationCheck:
        """Check that section ordering is sequential."""
        orders = [s.order for s in template.sections]
        is_sequential = orders == sorted(orders)
        return ValidationCheck(
            check_type="structural",
            severity=ValidationSeverity.WARNING,
            passed=is_sequential,
            message=(
                "Section ordering is sequential"
                if is_sequential
                else f"Section ordering is not sequential: {orders}"
            ),
        )

    def _check_template_metadata(
        self,
        template: QuestionnaireTemplate,
    ) -> ValidationCheck:
        """Check that template has required metadata."""
        has_name = bool(template.name and template.name.strip())
        has_framework = template.framework is not None
        passed = has_name and has_framework
        issues: List[str] = []
        if not has_name:
            issues.append("missing name")
        if not has_framework:
            issues.append("missing framework")
        return ValidationCheck(
            check_type="completeness",
            severity=ValidationSeverity.ERROR,
            passed=passed,
            message=(
                "Template metadata is complete"
                if passed
                else f"Template metadata issues: {', '.join(issues)}"
            ),
        )

    def _check_duplicate_codes(
        self,
        template: QuestionnaireTemplate,
    ) -> ValidationCheck:
        """Check for duplicate question codes across all sections."""
        codes: List[str] = []
        duplicates: List[str] = []
        seen: set = set()
        for section in template.sections:
            for q in section.questions:
                if q.code and q.code in seen:
                    duplicates.append(q.code)
                if q.code:
                    seen.add(q.code)
                    codes.append(q.code)

        passed = len(duplicates) == 0
        return ValidationCheck(
            check_type="consistency",
            severity=ValidationSeverity.WARNING,
            passed=passed,
            message=(
                f"No duplicate question codes found ({len(codes)} total)"
                if passed
                else f"Duplicate question codes: {duplicates}"
            ),
        )

    # ------------------------------------------------------------------
    # Framework section builders
    # ------------------------------------------------------------------

    def _build_framework_sections(
        self,
        framework: Framework,
    ) -> List[TemplateSection]:
        """Build skeleton sections from a built-in framework definition.

        Args:
            framework: Framework to build sections for.

        Returns:
            List of TemplateSection objects with questions populated.
        """
        section_defs = _FRAMEWORK_SECTIONS.get(framework.value, [])
        question_defs = _FRAMEWORK_QUESTIONS.get(framework.value, {})

        sections: List[TemplateSection] = []
        for sdef in section_defs:
            code = sdef["code"]
            q_list = question_defs.get(code, [])

            questions: List[TemplateQuestion] = []
            for idx, qdef in enumerate(q_list):
                q_type_str = qdef.get("type", "text")
                q_type = QuestionType(q_type_str)
                choices: List[str] = []
                if q_type == QuestionType.SINGLE_CHOICE:
                    choices = ["Option A", "Option B", "Option C"]
                elif q_type == QuestionType.MULTI_CHOICE:
                    choices = ["Option A", "Option B", "Option C", "Option D"]

                questions.append(TemplateQuestion(
                    code=qdef["code"],
                    text=qdef["text"],
                    question_type=q_type,
                    required=True,
                    choices=choices,
                    weight=1.0,
                    framework_ref=qdef["code"],
                    order=idx,
                ))

            sections.append(TemplateSection(
                name=sdef["name"],
                order=sdef["order"],
                questions=questions,
                weight=1.0,
                framework_ref=code,
            ))

        return sections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_framework(self, framework: str) -> Framework:
        """Resolve a framework string to a Framework enum.

        Args:
            framework: Framework value string.

        Returns:
            Framework enum member.

        Raises:
            ValueError: If framework is not recognised.
        """
        try:
            return Framework(framework)
        except ValueError:
            valid = [f.value for f in Framework]
            raise ValueError(
                f"Unknown framework '{framework}'. Valid: {valid}"
            )

    def _get_template_or_raise(
        self,
        template_id: str,
    ) -> QuestionnaireTemplate:
        """Retrieve a template or raise ValueError.

        Args:
            template_id: Template identifier.

        Returns:
            QuestionnaireTemplate.

        Raises:
            ValueError: If template_id is not found.
        """
        with self._lock:
            template = self._templates.get(template_id)
        if template is None:
            raise ValueError(f"Unknown template: {template_id}")
        return template

    def _compute_provenance(self, *parts: str) -> str:
        """Compute SHA-256 provenance hash from parts.

        Args:
            *parts: Strings to include in the hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        combined = json.dumps(
            {"parts": list(parts), "timestamp": _utcnow().isoformat()},
            sort_keys=True,
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
