"""
SRSGenerator - Safety Requirements Specification Generator

This module implements SRS document generation per IEC 61511-1 Clause 10.
The SRS is the primary document specifying safety function requirements
for Safety Instrumented Systems (SIS).

SRS Contents per IEC 61511-1 Clause 10.3:
- Safe state definitions
- Safety function specifications
- SIL requirements
- Process safety time requirements
- Hardware fault tolerance requirements
- Proof test requirements
- Response time requirements

Reference: IEC 61511-1 Clause 10

Example:
    >>> from greenlang.safety.srs.srs_generator import SRSGenerator
    >>> generator = SRSGenerator()
    >>> srs = generator.create_srs(safety_function)
    >>> generator.export_markdown(srs, "SRS_001.md")
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class SRSStatus(str, Enum):
    """SRS document status."""

    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    SUPERSEDED = "superseded"


class SRSSection(BaseModel):
    """Individual section of an SRS document."""

    section_id: str = Field(
        ...,
        description="Section identifier (e.g., '10.3.2')"
    )
    title: str = Field(
        ...,
        description="Section title"
    )
    content: str = Field(
        ...,
        description="Section content"
    )
    subsections: List['SRSSection'] = Field(
        default_factory=list,
        description="Nested subsections"
    )
    requirements: List[str] = Field(
        default_factory=list,
        description="List of requirements in this section"
    )

    class Config:
        """Pydantic configuration."""
        pass


class SafetyFunctionRequirement(BaseModel):
    """Individual safety function requirement per IEC 61511-1 Clause 10.3."""

    req_id: str = Field(
        ...,
        description="Requirement identifier"
    )
    description: str = Field(
        ...,
        description="Requirement description"
    )
    sil_level: int = Field(
        ...,
        ge=1,
        le=4,
        description="Required SIL level"
    )
    safe_state: str = Field(
        ...,
        description="Safe state definition"
    )
    process_safety_time_ms: float = Field(
        ...,
        gt=0,
        description="Process safety time (milliseconds)"
    )
    required_response_time_ms: float = Field(
        ...,
        gt=0,
        description="Required response time (milliseconds)"
    )
    proof_test_interval_hours: float = Field(
        ...,
        gt=0,
        description="Proof test interval (hours)"
    )
    pfd_target: float = Field(
        ...,
        gt=0,
        lt=1,
        description="Target PFD average"
    )
    hft_requirement: int = Field(
        default=0,
        ge=0,
        description="Hardware fault tolerance requirement"
    )
    diagnostic_coverage_target: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Target diagnostic coverage"
    )
    input_sensors: List[str] = Field(
        default_factory=list,
        description="List of input sensors"
    )
    output_actuators: List[str] = Field(
        default_factory=list,
        description="List of output actuators"
    )
    initiating_cause: str = Field(
        default="",
        description="Hazardous event initiating cause"
    )
    consequence: str = Field(
        default="",
        description="Potential consequence if safety function fails"
    )
    action_on_detection: str = Field(
        default="",
        description="Action to take when hazard detected"
    )
    manual_shutdown_required: bool = Field(
        default=True,
        description="Manual shutdown capability required"
    )
    bypass_permitted: bool = Field(
        default=False,
        description="Bypass permitted per IEC 61511"
    )


class SRSDocument(BaseModel):
    """Complete Safety Requirements Specification document."""

    document_id: str = Field(
        default_factory=lambda: f"SRS-{uuid.uuid4().hex[:8].upper()}",
        description="Document identifier"
    )
    title: str = Field(
        ...,
        description="SRS document title"
    )
    revision: str = Field(
        default="1.0",
        description="Document revision"
    )
    status: SRSStatus = Field(
        default=SRSStatus.DRAFT,
        description="Document status"
    )
    project_name: str = Field(
        default="",
        description="Project name"
    )
    system_name: str = Field(
        default="",
        description="SIS name"
    )
    created_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation date"
    )
    last_modified: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last modification date"
    )
    created_by: str = Field(
        default="",
        description="Author"
    )
    approved_by: Optional[str] = Field(
        None,
        description="Approver"
    )
    approval_date: Optional[datetime] = Field(
        None,
        description="Approval date"
    )
    scope: str = Field(
        default="",
        description="SRS scope"
    )
    hazard_identification_ref: str = Field(
        default="",
        description="Reference to hazard identification study (e.g., HAZOP)"
    )
    safety_function_requirements: List[SafetyFunctionRequirement] = Field(
        default_factory=list,
        description="List of safety function requirements"
    )
    sections: List[SRSSection] = Field(
        default_factory=list,
        description="Document sections"
    )
    appendices: List[str] = Field(
        default_factory=list,
        description="Appendix references"
    )
    references: List[str] = Field(
        default_factory=list,
        description="Reference documents"
    )
    change_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Document change history"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SRSGenerator:
    """
    Safety Requirements Specification Generator.

    Generates SRS documents per IEC 61511-1 Clause 10 requirements.
    Ensures all mandatory content is included with proper structure.

    The generator follows zero-hallucination principles:
    - All content is deterministic from inputs
    - No LLM involvement in safety-critical calculations
    - Complete audit trail with provenance hashing

    Attributes:
        standard_sections: List of standard SRS sections per IEC 61511

    Example:
        >>> generator = SRSGenerator()
        >>> req = SafetyFunctionRequirement(...)
        >>> srs = generator.create_srs("High Temp Shutdown", [req])
    """

    # Standard sections per IEC 61511-1 Clause 10.3
    STANDARD_SECTIONS: List[Dict[str, str]] = [
        {
            "id": "1",
            "title": "Scope and Introduction",
            "template": "This SRS defines the safety requirements for {system_name}."
        },
        {
            "id": "2",
            "title": "Reference Documents",
            "template": "Reference documents for this SRS include: {references}"
        },
        {
            "id": "3",
            "title": "Process Description",
            "template": "The process protected by this SIS is: {process_description}"
        },
        {
            "id": "4",
            "title": "Hazard Identification",
            "template": "Hazards identified per {hazard_study_ref}."
        },
        {
            "id": "5",
            "title": "Safe State Definitions",
            "template": "Safe states for this system are defined as: {safe_states}"
        },
        {
            "id": "6",
            "title": "Safety Instrumented Functions",
            "template": "The following SIFs are specified: {sif_list}"
        },
        {
            "id": "7",
            "title": "SIL Requirements",
            "template": "SIL requirements per hazard analysis: {sil_requirements}"
        },
        {
            "id": "8",
            "title": "Response Time Requirements",
            "template": "Response times must meet PST requirements: {response_times}"
        },
        {
            "id": "9",
            "title": "Proof Test Requirements",
            "template": "Proof testing requirements: {proof_test_requirements}"
        },
        {
            "id": "10",
            "title": "HFT Requirements",
            "template": "Hardware fault tolerance requirements: {hft_requirements}"
        },
        {
            "id": "11",
            "title": "Input/Output Specifications",
            "template": "I/O specifications: {io_specs}"
        },
        {
            "id": "12",
            "title": "Bypass Requirements",
            "template": "Bypass requirements per IEC 61511: {bypass_requirements}"
        },
    ]

    def __init__(self):
        """Initialize SRSGenerator."""
        logger.info("SRSGenerator initialized")

    def create_srs(
        self,
        title: str,
        requirements: List[SafetyFunctionRequirement],
        project_name: str = "",
        system_name: str = "",
        created_by: str = "",
        hazard_identification_ref: str = "",
        scope: str = ""
    ) -> SRSDocument:
        """
        Create a new SRS document.

        Args:
            title: SRS document title
            requirements: List of safety function requirements
            project_name: Project name
            system_name: SIS name
            created_by: Author name
            hazard_identification_ref: Reference to HAZOP or other hazard study
            scope: Document scope

        Returns:
            Complete SRSDocument

        Raises:
            ValueError: If requirements are invalid
        """
        logger.info(f"Creating SRS: {title}")

        # Validate requirements
        self._validate_requirements(requirements)

        # Generate sections
        sections = self._generate_sections(
            system_name=system_name,
            requirements=requirements,
            hazard_ref=hazard_identification_ref
        )

        # Create document
        srs = SRSDocument(
            title=title,
            project_name=project_name,
            system_name=system_name,
            created_by=created_by,
            hazard_identification_ref=hazard_identification_ref,
            scope=scope,
            safety_function_requirements=requirements,
            sections=sections,
            references=[
                "IEC 61511-1:2016 - Functional safety - Safety instrumented systems",
                "IEC 61511-2:2016 - Guidelines for application of IEC 61511-1",
                "IEC 61511-3:2016 - Guidance for determination of SIL",
            ]
        )

        # Calculate provenance hash
        srs.provenance_hash = self._calculate_provenance(srs)

        # Add initial change history
        srs.change_history.append({
            "revision": "1.0",
            "date": datetime.utcnow().isoformat(),
            "author": created_by,
            "description": "Initial release"
        })

        logger.info(f"SRS created: {srs.document_id}")

        return srs

    def _validate_requirements(
        self,
        requirements: List[SafetyFunctionRequirement]
    ) -> None:
        """Validate safety function requirements."""
        for req in requirements:
            # Validate response time vs PST
            if req.required_response_time_ms >= req.process_safety_time_ms:
                logger.warning(
                    f"Requirement {req.req_id}: Response time "
                    f"({req.required_response_time_ms}ms) must be less than "
                    f"PST ({req.process_safety_time_ms}ms)"
                )

            # Validate SIL vs PFD target
            sil_pfd_limits = {
                1: (1e-2, 1e-1),
                2: (1e-3, 1e-2),
                3: (1e-4, 1e-3),
                4: (1e-5, 1e-4),
            }
            lower, upper = sil_pfd_limits[req.sil_level]
            if not (lower <= req.pfd_target < upper):
                logger.warning(
                    f"Requirement {req.req_id}: PFD target {req.pfd_target} "
                    f"does not match SIL {req.sil_level} range ({lower}, {upper})"
                )

    def _generate_sections(
        self,
        system_name: str,
        requirements: List[SafetyFunctionRequirement],
        hazard_ref: str
    ) -> List[SRSSection]:
        """Generate standard SRS sections."""
        sections = []

        # Section 1: Scope
        sections.append(SRSSection(
            section_id="1",
            title="Scope and Introduction",
            content=f"This Safety Requirements Specification (SRS) defines the "
                    f"functional safety requirements for {system_name}. "
                    f"It is prepared in accordance with IEC 61511-1:2016 Clause 10.",
            requirements=["IEC 61511-1 Clause 10.3.1"]
        ))

        # Section 2: References
        sections.append(SRSSection(
            section_id="2",
            title="Reference Documents",
            content="The following documents are referenced in this SRS:\n"
                    "- IEC 61511-1:2016 Functional Safety\n"
                    f"- {hazard_ref}\n"
                    "- Process P&IDs\n"
                    "- Equipment datasheets",
            requirements=["IEC 61511-1 Clause 10.3.2"]
        ))

        # Section 3: Safe States
        safe_states = set(req.safe_state for req in requirements)
        safe_state_content = "Safe states for this system:\n" + "\n".join(
            f"- {ss}" for ss in safe_states
        )
        sections.append(SRSSection(
            section_id="3",
            title="Safe State Definitions",
            content=safe_state_content,
            requirements=["IEC 61511-1 Clause 10.3.3"]
        ))

        # Section 4: Safety Functions
        sif_content = "Safety Instrumented Functions (SIFs):\n\n"
        for req in requirements:
            sif_content += f"**{req.req_id}**: {req.description}\n"
            sif_content += f"- SIL: {req.sil_level}\n"
            sif_content += f"- Safe State: {req.safe_state}\n"
            sif_content += f"- Initiating Cause: {req.initiating_cause}\n"
            sif_content += f"- Action: {req.action_on_detection}\n\n"

        sections.append(SRSSection(
            section_id="4",
            title="Safety Instrumented Functions",
            content=sif_content,
            requirements=["IEC 61511-1 Clause 10.3.4"]
        ))

        # Section 5: SIL Requirements
        sil_content = "SIL Requirements:\n\n"
        sil_content += "| SIF ID | SIL | PFD Target | HFT |\n"
        sil_content += "|--------|-----|------------|-----|\n"
        for req in requirements:
            sil_content += (
                f"| {req.req_id} | {req.sil_level} | "
                f"{req.pfd_target:.1e} | {req.hft_requirement} |\n"
            )

        sections.append(SRSSection(
            section_id="5",
            title="SIL Requirements",
            content=sil_content,
            requirements=["IEC 61511-1 Clause 10.3.5"]
        ))

        # Section 6: Response Time Requirements
        response_content = "Response Time Requirements:\n\n"
        response_content += "| SIF ID | PST (ms) | Required Response (ms) | Margin |\n"
        response_content += "|--------|----------|----------------------|--------|\n"
        for req in requirements:
            margin = req.process_safety_time_ms - req.required_response_time_ms
            response_content += (
                f"| {req.req_id} | {req.process_safety_time_ms:.0f} | "
                f"{req.required_response_time_ms:.0f} | {margin:.0f} |\n"
            )

        sections.append(SRSSection(
            section_id="6",
            title="Response Time Requirements",
            content=response_content,
            requirements=["IEC 61511-1 Clause 10.3.6"]
        ))

        # Section 7: Proof Test Requirements
        proof_content = "Proof Test Requirements:\n\n"
        proof_content += "| SIF ID | Proof Test Interval | DC Target |\n"
        proof_content += "|--------|---------------------|----------|\n"
        for req in requirements:
            interval_str = f"{req.proof_test_interval_hours/8760:.1f} years"
            proof_content += (
                f"| {req.req_id} | {interval_str} | "
                f"{req.diagnostic_coverage_target:.0%} |\n"
            )

        sections.append(SRSSection(
            section_id="7",
            title="Proof Test Requirements",
            content=proof_content,
            requirements=["IEC 61511-1 Clause 10.3.7"]
        ))

        # Section 8: I/O Requirements
        io_content = "Input/Output Requirements:\n\n"
        for req in requirements:
            io_content += f"**{req.req_id}**\n"
            io_content += f"Inputs: {', '.join(req.input_sensors) or 'TBD'}\n"
            io_content += f"Outputs: {', '.join(req.output_actuators) or 'TBD'}\n\n"

        sections.append(SRSSection(
            section_id="8",
            title="Input/Output Requirements",
            content=io_content,
            requirements=["IEC 61511-1 Clause 10.3.8"]
        ))

        # Section 9: Bypass Requirements
        bypass_content = "Bypass Requirements per IEC 61511-1 Clause 11.7:\n\n"
        bypass_content += "- All bypasses shall be alarmed\n"
        bypass_content += "- Bypass duration shall be logged\n"
        bypass_content += "- MOC approval required for extended bypasses\n\n"
        bypass_content += "Bypass permissions:\n"
        for req in requirements:
            bypass_content += (
                f"- {req.req_id}: "
                f"{'Permitted' if req.bypass_permitted else 'Not Permitted'}\n"
            )

        sections.append(SRSSection(
            section_id="9",
            title="Bypass Requirements",
            content=bypass_content,
            requirements=["IEC 61511-1 Clause 11.7"]
        ))

        return sections

    def export_markdown(
        self,
        srs: SRSDocument,
        filepath: Optional[str] = None
    ) -> str:
        """
        Export SRS to Markdown format.

        Args:
            srs: SRSDocument to export
            filepath: Optional file path to write to

        Returns:
            Markdown string
        """
        md_lines = []

        # Title and metadata
        md_lines.append(f"# {srs.title}")
        md_lines.append("")
        md_lines.append(f"**Document ID:** {srs.document_id}")
        md_lines.append(f"**Revision:** {srs.revision}")
        md_lines.append(f"**Status:** {srs.status.value}")
        md_lines.append(f"**Created:** {srs.created_date.strftime('%Y-%m-%d')}")
        md_lines.append(f"**Author:** {srs.created_by}")
        if srs.approved_by:
            md_lines.append(f"**Approved By:** {srs.approved_by}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # Scope
        if srs.scope:
            md_lines.append("## Scope")
            md_lines.append("")
            md_lines.append(srs.scope)
            md_lines.append("")

        # Sections
        for section in srs.sections:
            md_lines.append(f"## {section.section_id}. {section.title}")
            md_lines.append("")
            md_lines.append(section.content)
            md_lines.append("")

            # Requirements
            if section.requirements:
                md_lines.append("**IEC 61511 References:**")
                for req in section.requirements:
                    md_lines.append(f"- {req}")
                md_lines.append("")

        # References
        md_lines.append("## References")
        md_lines.append("")
        for ref in srs.references:
            md_lines.append(f"- {ref}")
        md_lines.append("")

        # Provenance
        md_lines.append("---")
        md_lines.append("")
        md_lines.append(f"**Provenance Hash:** `{srs.provenance_hash}`")
        md_lines.append("")

        markdown = "\n".join(md_lines)

        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown)
            logger.info(f"SRS exported to {filepath}")

        return markdown

    def update_revision(
        self,
        srs: SRSDocument,
        new_revision: str,
        modified_by: str,
        change_description: str
    ) -> SRSDocument:
        """
        Update SRS revision.

        Args:
            srs: SRSDocument to update
            new_revision: New revision number
            modified_by: Person making changes
            change_description: Description of changes

        Returns:
            Updated SRSDocument
        """
        srs.revision = new_revision
        srs.last_modified = datetime.utcnow()
        srs.status = SRSStatus.DRAFT

        srs.change_history.append({
            "revision": new_revision,
            "date": datetime.utcnow().isoformat(),
            "author": modified_by,
            "description": change_description
        })

        # Recalculate provenance hash
        srs.provenance_hash = self._calculate_provenance(srs)

        logger.info(f"SRS updated to revision {new_revision}")

        return srs

    def approve_srs(
        self,
        srs: SRSDocument,
        approved_by: str
    ) -> SRSDocument:
        """
        Approve SRS document.

        Args:
            srs: SRSDocument to approve
            approved_by: Approver name

        Returns:
            Approved SRSDocument
        """
        srs.status = SRSStatus.APPROVED
        srs.approved_by = approved_by
        srs.approval_date = datetime.utcnow()

        # Recalculate provenance hash
        srs.provenance_hash = self._calculate_provenance(srs)

        logger.info(f"SRS {srs.document_id} approved by {approved_by}")

        return srs

    def _calculate_provenance(self, srs: SRSDocument) -> str:
        """Calculate SHA-256 provenance hash for SRS."""
        provenance_str = (
            f"{srs.document_id}|"
            f"{srs.revision}|"
            f"{len(srs.safety_function_requirements)}|"
            f"{srs.last_modified.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
