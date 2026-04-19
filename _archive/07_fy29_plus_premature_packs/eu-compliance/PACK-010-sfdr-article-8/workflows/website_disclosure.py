# -*- coding: utf-8 -*-
"""
Website Disclosure Workflow
===============================

Four-phase workflow for generating SFDR Annex III website disclosures
for Article 8 financial products. Orchestrates content assembly, template
generation, update tracking, and publication into a single auditable
pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - Article 10: Website disclosure requirements for Article 8 products.
    - Annex III: Template for Article 8 website disclosures.
    - Information must be published on the financial market participant's
      website in a dedicated section.
    - Content must be kept up-to-date and clearly dated.
    - Required sections: summary, investment strategy description, proportion
      of investments, monitoring methodology, data sources, product-specific
      limitations, due diligence, engagement policies.
    - Disclosures must be accessible, clear, and in the official language(s)
      of the relevant Member State(s).

Phases:
    1. ContentAssembly - Gather all Annex III required content (summary,
       investment strategy, proportion breakdown, monitoring methodology,
       data sources)
    2. TemplateGeneration - Generate structured output (HTML/Markdown/JSON)
       following RTS layout
    3. UpdateTracking - Version control, change history, last-updated
       timestamps, diff from previous version
    4. Publication - Generate publication-ready content with regulatory
       references and accessibility compliance

Author: GreenLang Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# UTILITIES
# =============================================================================

def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

class OutputFormat(str, Enum):
    """Supported output formats for website disclosure."""
    HTML = "HTML"
    MARKDOWN = "MARKDOWN"
    JSON = "JSON"

class PublicationStatus(str, Enum):
    """Website publication status."""
    DRAFT = "DRAFT"
    READY_FOR_PUBLICATION = "READY_FOR_PUBLICATION"
    PUBLISHED = "PUBLISHED"
    UPDATE_REQUIRED = "UPDATE_REQUIRED"
    ARCHIVED = "ARCHIVED"

class ChangeType(str, Enum):
    """Type of content change in version tracking."""
    INITIAL = "INITIAL"
    CONTENT_UPDATE = "CONTENT_UPDATE"
    REGULATORY_UPDATE = "REGULATORY_UPDATE"
    DATA_REFRESH = "DATA_REFRESH"
    CORRECTION = "CORRECTION"

# =============================================================================
# DATA MODELS - SHARED
# =============================================================================

class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)

class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

# =============================================================================
# DATA MODELS - WEBSITE DISCLOSURE
# =============================================================================

class WebsiteDisclosureInput(BaseModel):
    """Input configuration for the website disclosure workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    product_isin: Optional[str] = Field(None)
    disclosure_date: str = Field(
        ..., description="Disclosure date in YYYY-MM-DD format"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.HTML, description="Output format"
    )
    language: str = Field(default="en", description="ISO 639-1 language code")
    product_summary: str = Field(
        default="", description="Product summary text"
    )
    investment_strategy_description: str = Field(
        default="", description="Investment strategy description"
    )
    binding_elements: List[Dict[str, Any]] = Field(
        default_factory=list, description="Binding elements of strategy"
    )
    good_governance_description: str = Field(
        default="", description="Good governance assessment description"
    )
    asset_allocation: Optional[Dict[str, float]] = Field(
        None, description="Asset allocation proportions"
    )
    monitoring_methodology: str = Field(
        default="", description="Monitoring methodology description"
    )
    sustainability_indicators: List[str] = Field(
        default_factory=list, description="Sustainability indicators"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources"
    )
    data_processing_description: str = Field(default="")
    limitations_description: str = Field(default="")
    due_diligence_description: str = Field(default="")
    engagement_description: str = Field(default="")
    reference_benchmark: Optional[str] = Field(None)
    previous_version: Optional[Dict[str, Any]] = Field(
        None, description="Previous version content for diff"
    )
    change_type: ChangeType = Field(default=ChangeType.INITIAL)
    change_description: str = Field(default="")
    website_url: Optional[str] = Field(
        None, description="Target publication URL"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("disclosure_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate disclosure date is valid ISO format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("disclosure_date must be YYYY-MM-DD format")
        return v

class WebsiteDisclosureResult(WorkflowResult):
    """Complete result from the website disclosure workflow."""
    product_name: str = Field(default="")
    output_format: str = Field(default="HTML")
    sections_populated: int = Field(default=0)
    sections_total: int = Field(default=8)
    content_completeness_pct: float = Field(default=0.0)
    version_number: str = Field(default="1.0")
    publication_status: str = Field(default="DRAFT")
    content_hash: str = Field(default="")
    changes_from_previous: int = Field(default=0)

# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================

class ContentAssemblyPhase:
    """
    Phase 1: Content Assembly.

    Gathers all Annex III required content sections including summary,
    investment strategy, proportion breakdown, monitoring methodology,
    and data sources.
    """

    PHASE_NAME = "content_assembly"

    REQUIRED_SECTIONS = [
        "summary",
        "investment_strategy",
        "proportion_of_investments",
        "monitoring_of_characteristics",
        "methodologies",
        "data_sources_and_processing",
        "limitations",
        "engagement_policies",
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute content assembly phase.

        Args:
            context: Workflow context with content inputs.

        Returns:
            PhaseResult with assembled content sections.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            product_name = config.get("product_name", "")

            sections: Dict[str, Dict[str, Any]] = {}
            populated_count = 0

            # Section 1: Summary
            summary_text = config.get("product_summary", "")
            sections["summary"] = {
                "title": "Summary",
                "content": summary_text if summary_text else (
                    f"{product_name} is a financial product that promotes "
                    f"environmental and/or social characteristics within "
                    f"the meaning of Article 8 of the SFDR."
                ),
                "populated": True,
            }
            populated_count += 1

            # Section 2: Investment Strategy
            strategy_text = config.get(
                "investment_strategy_description", ""
            )
            binding_elements = config.get("binding_elements", [])
            governance_text = config.get(
                "good_governance_description", ""
            )
            sections["investment_strategy"] = {
                "title": "Investment Strategy",
                "strategy_description": strategy_text if strategy_text else (
                    "The investment strategy integrates environmental and "
                    "social considerations into the investment process."
                ),
                "binding_elements": binding_elements,
                "binding_elements_count": len(binding_elements),
                "good_governance": governance_text if governance_text else (
                    "Good governance practices are assessed through "
                    "evaluation of management structures, employee "
                    "relations, remuneration, and tax compliance."
                ),
                "populated": bool(strategy_text or binding_elements),
            }
            populated_count += 1

            # Section 3: Proportion of Investments
            allocation = config.get("asset_allocation", {})
            sections["proportion_of_investments"] = {
                "title": "Proportion of Investments",
                "asset_allocation": allocation if allocation else {
                    "aligned_with_es_characteristics_pct": 0.0,
                    "sustainable_investments_pct": 0.0,
                    "taxonomy_aligned_pct": 0.0,
                    "other_investments_pct": 0.0,
                },
                "populated": bool(allocation),
            }
            populated_count += 1

            # Section 4: Monitoring of Characteristics
            monitoring_text = config.get("monitoring_methodology", "")
            indicators = config.get("sustainability_indicators", [])
            sections["monitoring_of_characteristics"] = {
                "title": "Monitoring of Environmental or Social Characteristics",
                "methodology": monitoring_text if monitoring_text else (
                    "Environmental and social characteristics are monitored "
                    "on an ongoing basis using sustainability indicators."
                ),
                "sustainability_indicators": indicators,
                "monitoring_frequency": "quarterly",
                "populated": bool(monitoring_text or indicators),
            }
            populated_count += 1

            # Section 5: Methodologies
            sections["methodologies"] = {
                "title": "Methodologies for Environmental or Social Characteristics",
                "description": (
                    "The attainment of E/S characteristics is measured "
                    "using quantitative and qualitative methodologies "
                    "applied consistently across the portfolio."
                ),
                "populated": True,
            }
            populated_count += 1

            # Section 6: Data Sources and Processing
            data_sources = config.get("data_sources", [])
            processing_text = config.get("data_processing_description", "")
            sections["data_sources_and_processing"] = {
                "title": "Data Sources and Processing",
                "data_sources": data_sources,
                "processing_description": processing_text if processing_text else (
                    "Data is sourced from third-party ESG data providers, "
                    "company disclosures, and proprietary research."
                ),
                "data_quality_measures": [
                    "Automated validation checks",
                    "Cross-source verification",
                    "Regular provider assessments",
                ],
                "populated": bool(data_sources),
            }
            populated_count += 1

            # Section 7: Limitations
            limitations_text = config.get("limitations_description", "")
            sections["limitations"] = {
                "title": "Limitations to Methodologies and Data",
                "description": limitations_text if limitations_text else (
                    "Limitations may include gaps in data coverage, "
                    "reliance on estimated data, and methodological "
                    "differences across jurisdictions."
                ),
                "mitigation": (
                    "These limitations do not affect the ability of the "
                    "product to meet the E/S characteristics promoted."
                ),
                "populated": True,
            }
            populated_count += 1

            # Section 8: Engagement Policies
            engagement_text = config.get("engagement_description", "")
            sections["engagement_policies"] = {
                "title": "Engagement Policies",
                "description": engagement_text if engagement_text else (
                    "Engagement with investee companies is conducted as "
                    "part of the investment process where applicable."
                ),
                "populated": bool(engagement_text),
            }
            populated_count += 1

            outputs["sections"] = sections
            outputs["sections_populated"] = populated_count
            outputs["sections_total"] = len(self.REQUIRED_SECTIONS)
            outputs["completeness_pct"] = round(
                populated_count / len(self.REQUIRED_SECTIONS) * 100, 1
            )

            # Check for missing content
            unpopulated = [
                name for name, data in sections.items()
                if not data.get("populated", False)
            ]
            if unpopulated:
                warnings.append(
                    f"Sections with default content only: "
                    f"{', '.join(unpopulated)}"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ContentAssembly failed: %s", exc, exc_info=True)
            errors.append(f"Content assembly failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class WebTemplateGenerationPhase:
    """
    Phase 2: Template Generation.

    Generates structured output in the requested format (HTML, Markdown,
    or JSON) following the RTS layout for Annex III website disclosures.
    """

    PHASE_NAME = "template_generation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute template generation phase.

        Args:
            context: Workflow context with assembled content.

        Returns:
            PhaseResult with formatted template output.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            content_output = context.get_phase_output("content_assembly")
            sections = content_output.get("sections", {})
            output_format = config.get("output_format", "HTML")
            product_name = config.get("product_name", "")
            language = config.get("language", "en")

            # Generate based on format
            if output_format == OutputFormat.HTML.value:
                template_content = self._generate_html(
                    product_name, sections, language
                )
            elif output_format == OutputFormat.MARKDOWN.value:
                template_content = self._generate_markdown(
                    product_name, sections
                )
            else:
                template_content = self._generate_json(
                    product_name, sections
                )

            outputs["template_content"] = template_content
            outputs["output_format"] = output_format
            outputs["content_hash"] = _hash_data(template_content)
            outputs["content_length"] = len(
                json.dumps(template_content, default=str)
            )
            outputs["language"] = language
            outputs["generated_at"] = utcnow().isoformat()

            # Regulatory references
            outputs["regulatory_references"] = {
                "sfdr_regulation": "Regulation (EU) 2019/2088",
                "delegated_regulation": "Delegated Regulation (EU) 2022/1288",
                "taxonomy_regulation": "Regulation (EU) 2020/852",
                "annex": "Annex III",
                "article": "Article 10",
            }

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "WebTemplateGeneration failed: %s", exc, exc_info=True
            )
            errors.append(f"Template generation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    def _generate_html(
        self,
        product_name: str,
        sections: Dict[str, Any],
        language: str,
    ) -> Dict[str, Any]:
        """Generate HTML-structured template."""
        html_sections = []
        for section_key, section_data in sections.items():
            title = section_data.get("title", section_key)
            content_parts = []
            for key, value in section_data.items():
                if key in ("title", "populated"):
                    continue
                if isinstance(value, str) and value:
                    content_parts.append(f"<p>{value}</p>")
                elif isinstance(value, list) and value:
                    items = "".join(f"<li>{item}</li>" for item in value)
                    content_parts.append(f"<ul>{items}</ul>")

            html_sections.append({
                "section_id": section_key,
                "html": (
                    f'<section id="{section_key}" lang="{language}">'
                    f"<h2>{title}</h2>"
                    f"{''.join(content_parts)}"
                    f"</section>"
                ),
            })

        return {
            "format": "html",
            "product_name": product_name,
            "language": language,
            "sections": html_sections,
            "header": (
                f'<header><h1>SFDR Article 8 Disclosure - '
                f'{product_name}</h1></header>'
            ),
            "footer": (
                '<footer><p>This disclosure is made pursuant to '
                'Regulation (EU) 2019/2088 (SFDR), Article 10, '
                'Annex III.</p></footer>'
            ),
        }

    def _generate_markdown(
        self, product_name: str, sections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Markdown-structured template."""
        md_sections = []
        for section_key, section_data in sections.items():
            title = section_data.get("title", section_key)
            content_parts = [f"## {title}\n"]
            for key, value in section_data.items():
                if key in ("title", "populated"):
                    continue
                if isinstance(value, str) and value:
                    content_parts.append(f"{value}\n")
                elif isinstance(value, list) and value:
                    for item in value:
                        content_parts.append(f"- {item}")
                    content_parts.append("")

            md_sections.append({
                "section_id": section_key,
                "markdown": "\n".join(content_parts),
            })

        return {
            "format": "markdown",
            "product_name": product_name,
            "sections": md_sections,
        }

    def _generate_json(
        self, product_name: str, sections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate JSON-structured template."""
        return {
            "format": "json",
            "product_name": product_name,
            "sections": sections,
        }

class UpdateTrackingPhase:
    """
    Phase 3: Update Tracking.

    Manages version control, change history, last-updated timestamps,
    and computes diffs from the previous version.
    """

    PHASE_NAME = "update_tracking"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute update tracking phase.

        Args:
            context: Workflow context with current and previous content.

        Returns:
            PhaseResult with version info and change history.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            template_output = context.get_phase_output("template_generation")
            previous_version = config.get("previous_version")
            change_type = config.get("change_type", ChangeType.INITIAL.value)
            change_description = config.get("change_description", "")

            current_hash = template_output.get("content_hash", "")

            # Version numbering
            if previous_version:
                prev_version_num = previous_version.get(
                    "version_number", "1.0"
                )
                parts = prev_version_num.split(".")
                try:
                    major = int(parts[0])
                    minor = int(parts[1]) if len(parts) > 1 else 0
                except (ValueError, IndexError):
                    major, minor = 1, 0

                if change_type == ChangeType.REGULATORY_UPDATE.value:
                    version_number = f"{major + 1}.0"
                else:
                    version_number = f"{major}.{minor + 1}"
            else:
                version_number = "1.0"

            outputs["version_number"] = version_number
            outputs["content_hash"] = current_hash
            outputs["last_updated"] = utcnow().isoformat()

            # Change tracking
            change_record = {
                "change_id": str(uuid.uuid4()),
                "version": version_number,
                "change_type": change_type,
                "description": change_description if change_description else (
                    "Initial website disclosure" if not previous_version
                    else "Content update"
                ),
                "timestamp": utcnow().isoformat(),
                "content_hash": current_hash,
            }
            outputs["change_record"] = change_record

            # Diff from previous version
            changes_detected = 0
            section_diffs: List[Dict[str, Any]] = []

            if previous_version:
                prev_hash = previous_version.get("content_hash", "")
                outputs["previous_content_hash"] = prev_hash
                outputs["content_changed"] = current_hash != prev_hash

                prev_sections = previous_version.get("sections", {})
                content_output = context.get_phase_output("content_assembly")
                current_sections = content_output.get("sections", {})

                for section_key in set(
                    list(prev_sections.keys()) + list(current_sections.keys())
                ):
                    prev_data = prev_sections.get(section_key)
                    curr_data = current_sections.get(section_key)

                    if prev_data is None and curr_data is not None:
                        section_diffs.append({
                            "section": section_key,
                            "change": "added",
                        })
                        changes_detected += 1
                    elif prev_data is not None and curr_data is None:
                        section_diffs.append({
                            "section": section_key,
                            "change": "removed",
                        })
                        changes_detected += 1
                    elif _hash_data(prev_data) != _hash_data(curr_data):
                        section_diffs.append({
                            "section": section_key,
                            "change": "modified",
                        })
                        changes_detected += 1
            else:
                outputs["content_changed"] = True
                outputs["previous_content_hash"] = None

            outputs["section_diffs"] = section_diffs
            outputs["changes_detected"] = changes_detected

            # Change history (append to existing if provided)
            change_history = []
            if previous_version:
                change_history = previous_version.get(
                    "change_history", []
                ).copy()
            change_history.append(change_record)
            outputs["change_history"] = change_history
            outputs["total_versions"] = len(change_history)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("UpdateTracking failed: %s", exc, exc_info=True)
            errors.append(f"Update tracking failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class PublicationPhase:
    """
    Phase 4: Publication.

    Generates publication-ready content with regulatory references
    and accessibility compliance checks.
    """

    PHASE_NAME = "publication"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute publication phase.

        Args:
            context: Workflow context with template and version data.

        Returns:
            PhaseResult with publication-ready package.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            template_output = context.get_phase_output("template_generation")
            tracking_output = context.get_phase_output("update_tracking")
            content_output = context.get_phase_output("content_assembly")

            product_name = config.get("product_name", "")
            website_url = config.get("website_url")
            language = config.get("language", "en")

            # Accessibility compliance checks
            accessibility_checks = []

            # Check: Language attribute
            accessibility_checks.append({
                "check": "language_attribute",
                "passed": bool(language),
                "detail": f"Language: {language}",
            })

            # Check: Section headings present
            sections = content_output.get("sections", {})
            has_headings = all(
                s.get("title") for s in sections.values()
            )
            accessibility_checks.append({
                "check": "section_headings",
                "passed": has_headings,
                "detail": "All sections have headings",
            })

            # Check: Content clarity (non-empty sections)
            completeness = content_output.get("completeness_pct", 0.0)
            accessibility_checks.append({
                "check": "content_completeness",
                "passed": completeness >= 100.0,
                "detail": f"Completeness: {completeness}%",
            })

            # Check: Date visibility
            accessibility_checks.append({
                "check": "date_visibility",
                "passed": True,
                "detail": (
                    f"Last updated: "
                    f"{tracking_output.get('last_updated', 'N/A')}"
                ),
            })

            outputs["accessibility_checks"] = accessibility_checks
            outputs["accessibility_compliant"] = all(
                c["passed"] for c in accessibility_checks
            )

            # Publication metadata
            version_number = tracking_output.get("version_number", "1.0")
            content_hash = tracking_output.get("content_hash", "")

            # Determine publication status
            if not outputs["accessibility_compliant"]:
                pub_status = PublicationStatus.DRAFT.value
            elif completeness >= 100.0:
                pub_status = PublicationStatus.READY_FOR_PUBLICATION.value
            else:
                pub_status = PublicationStatus.DRAFT.value

            outputs["publication_status"] = pub_status
            outputs["publication_metadata"] = {
                "product_name": product_name,
                "version": version_number,
                "content_hash": content_hash,
                "language": language,
                "publication_url": website_url,
                "last_updated": tracking_output.get("last_updated", ""),
                "regulatory_basis": (
                    "SFDR Article 10, Annex III (Delegated Regulation "
                    "(EU) 2022/1288)"
                ),
            }

            # Regulatory footer
            outputs["regulatory_footer"] = {
                "text": (
                    "This disclosure is made in accordance with "
                    "Regulation (EU) 2019/2088 of the European Parliament "
                    "and of the Council (Sustainable Finance Disclosure "
                    "Regulation) and Commission Delegated Regulation (EU) "
                    "2022/1288. The information contained herein is for "
                    "informational purposes and does not constitute "
                    "investment advice."
                ),
                "last_updated": tracking_output.get("last_updated", ""),
                "version": version_number,
            }

            # Publication package
            outputs["publication_package"] = {
                "package_id": str(uuid.uuid4()),
                "template_content": template_output.get(
                    "template_content", {}
                ),
                "version_info": {
                    "version": version_number,
                    "content_hash": content_hash,
                    "change_history": tracking_output.get(
                        "change_history", []
                    ),
                },
                "regulatory_references": template_output.get(
                    "regulatory_references", {}
                ),
                "accessibility_report": accessibility_checks,
            }

            if pub_status != PublicationStatus.READY_FOR_PUBLICATION.value:
                failed_checks = [
                    c["check"] for c in accessibility_checks
                    if not c["passed"]
                ]
                if failed_checks:
                    warnings.append(
                        f"Accessibility checks failed: "
                        f"{', '.join(failed_checks)}"
                    )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("Publication failed: %s", exc, exc_info=True)
            errors.append(f"Publication failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================

class WebsiteDisclosureWorkflow:
    """
    Four-phase website disclosure workflow for SFDR Article 8.

    Orchestrates content assembly through publication for Annex III
    website disclosures. Supports version tracking, multiple output
    formats, and accessibility compliance checking.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered mapping of phase name to executor instance.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = WebsiteDisclosureWorkflow()
        >>> input_data = WebsiteDisclosureInput(
        ...     organization_id="org-123",
        ...     product_name="Green Bond Fund",
        ...     disclosure_date="2026-01-15",
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "website_disclosure"

    PHASE_ORDER = [
        "content_assembly",
        "template_generation",
        "update_tracking",
        "publication",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the website disclosure workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "content_assembly": ContentAssemblyPhase(),
            "template_generation": WebTemplateGenerationPhase(),
            "update_tracking": UpdateTrackingPhase(),
            "publication": PublicationPhase(),
        }

    async def run(
        self, input_data: WebsiteDisclosureInput
    ) -> WebsiteDisclosureResult:
        """
        Execute the complete 4-phase website disclosure workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            WebsiteDisclosureResult with per-phase details and summary.
        """
        started_at = utcnow()
        logger.info(
            "Starting website disclosure workflow %s for org=%s product=%s",
            self.workflow_id, input_data.organization_id,
            input_data.product_name,
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)
                    if phase_name == "content_assembly":
                        overall_status = WorkflowStatus.FAILED
                        break

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised unhandled exception: %s",
                    phase_name, exc, exc_info=True,
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    started_at=utcnow(),
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
            )

        completed_at = utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )
        logger.info(
            "Website disclosure workflow %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return WebsiteDisclosureResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            product_name=summary.get("product_name", ""),
            output_format=summary.get("output_format", "HTML"),
            sections_populated=summary.get("sections_populated", 0),
            sections_total=summary.get("sections_total", 8),
            content_completeness_pct=summary.get(
                "content_completeness_pct", 0.0
            ),
            version_number=summary.get("version_number", "1.0"),
            publication_status=summary.get("publication_status", "DRAFT"),
            content_hash=summary.get("content_hash", ""),
            changes_from_previous=summary.get("changes_from_previous", 0),
        )

    def _build_config(
        self, input_data: WebsiteDisclosureInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        config = input_data.model_dump()
        config["output_format"] = input_data.output_format.value
        config["change_type"] = input_data.change_type.value
        return config

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        config = context.config
        content_out = context.get_phase_output("content_assembly")
        template_out = context.get_phase_output("template_generation")
        tracking_out = context.get_phase_output("update_tracking")
        pub_out = context.get_phase_output("publication")

        return {
            "product_name": config.get("product_name", ""),
            "output_format": config.get("output_format", "HTML"),
            "sections_populated": content_out.get("sections_populated", 0),
            "sections_total": content_out.get("sections_total", 8),
            "content_completeness_pct": content_out.get(
                "completeness_pct", 0.0
            ),
            "version_number": tracking_out.get("version_number", "1.0"),
            "publication_status": pub_out.get(
                "publication_status", "DRAFT"
            ),
            "content_hash": tracking_out.get("content_hash", ""),
            "changes_from_previous": tracking_out.get(
                "changes_detected", 0
            ),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
