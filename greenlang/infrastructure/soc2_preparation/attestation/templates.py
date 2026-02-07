# -*- coding: utf-8 -*-
"""
Attestation Templates Module - SEC-009 Phase 7

Provides pre-built attestation document templates for SOC 2 Type II audits.
Supports template variable substitution and PDF export for signature collection.

Template Types:
    - soc2_readiness_attestation: CEO/CISO sign pre-audit
    - management_assertion_letter: CEO/CFO sign with report
    - control_owner_attestation: Control owners quarterly
    - subservice_organization_list: List of carved-out services
    - complementary_user_entity_controls: Customer responsibilities

Classes:
    - TemplateType: Enumeration of supported template types
    - Document: Generated document model
    - AttestationTemplates: Template generation and population

Example:
    >>> templates = AttestationTemplates()
    >>> variables = {
    ...     "company_name": "GreenLang Inc.",
    ...     "audit_period": "January 1, 2026 - December 31, 2026",
    ...     "ceo_name": "Jane Smith",
    ... }
    >>> doc = templates.generate_document("soc2_readiness_attestation", variables)
    >>> pdf_bytes = templates.export_pdf(doc)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TemplateType(str, Enum):
    """Supported attestation template types for SOC 2 audits."""

    SOC2_READINESS_ATTESTATION = "soc2_readiness_attestation"
    """CEO/CISO sign pre-audit to attest readiness for SOC 2 examination."""

    MANAGEMENT_ASSERTION_LETTER = "management_assertion_letter"
    """CEO/CFO sign with report to assert control effectiveness."""

    CONTROL_OWNER_ATTESTATION = "control_owner_attestation"
    """Control owners sign quarterly to confirm control operation."""

    SUBSERVICE_ORGANIZATION_LIST = "subservice_organization_list"
    """List of carved-out services with management acknowledgment."""

    COMPLEMENTARY_USER_ENTITY_CONTROLS = "complementary_user_entity_controls"
    """Customer responsibilities that complement service organization controls."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """Generated attestation document with content and metadata.

    Attributes:
        document_id: Unique identifier for the document.
        template_type: Type of template used to generate this document.
        title: Document title.
        content: Generated document content (Markdown format).
        content_hash: SHA-256 hash of the content for integrity verification.
        variables_used: Dictionary of variables substituted in the template.
        generated_at: Timestamp when the document was generated (UTC).
        generated_by: User ID of the person who generated the document.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the document.",
    )
    template_type: TemplateType = Field(
        ...,
        description="Type of template used to generate this document.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Document title.",
    )
    content: str = Field(
        ...,
        description="Generated document content (Markdown format).",
    )
    content_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash of the content for integrity verification.",
    )
    variables_used: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of variables substituted in the template.",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the document was generated (UTC).",
    )
    generated_by: str = Field(
        default="",
        max_length=256,
        description="User ID of the person who generated the document.",
    )


# ---------------------------------------------------------------------------
# Template Content
# ---------------------------------------------------------------------------


# Template dictionary with Markdown content
_TEMPLATES: Dict[TemplateType, str] = {
    TemplateType.SOC2_READINESS_ATTESTATION: """
# SOC 2 Type II Readiness Attestation

## {{company_name}}

**Document ID:** {{document_id}}
**Date:** {{current_date}}
**Audit Period:** {{audit_period}}

---

## Management Attestation of SOC 2 Type II Readiness

We, the undersigned officers of {{company_name}} ("the Company"), hereby attest
to the following statements regarding our organization's readiness for the
SOC 2 Type II examination:

### 1. Control Environment

We affirm that the Company has established and maintains a comprehensive system
of internal controls designed to meet the Trust Services Criteria for:

{{#tsc_categories}}
- {{category_name}}
{{/tsc_categories}}

### 2. Control Design and Implementation

We confirm that:

- All controls identified in our control matrix have been designed appropriately
  to meet the stated criteria.
- Controls have been implemented and are operating as designed as of {{attestation_date}}.
- Control owners have been assigned and have acknowledged their responsibilities.

### 3. Risk Assessment

We attest that:

- A formal risk assessment has been conducted within the past 12 months.
- Identified risks have been evaluated and appropriate controls have been
  implemented to mitigate those risks.
- The risk register is maintained and reviewed quarterly.

### 4. Evidence and Documentation

We confirm that:

- Sufficient evidence has been collected to demonstrate control effectiveness.
- Documentation is complete, accurate, and available for auditor review.
- Evidence retention policies meet SOC 2 requirements (minimum 7 years).

### 5. Remediation Status

We attest that:

- No material deficiencies exist in the control environment.
- All previously identified findings have been remediated or have approved
  remediation plans in place.
- Remediation activities are tracked and monitored.

### 6. Subservice Organizations

We confirm that:

- All subservice organizations have been identified and documented.
- Appropriate oversight and monitoring is in place for subservice organizations.
- CUEC (Complementary User Entity Controls) have been communicated to customers.

---

## Signatures

**Chief Executive Officer**

Name: {{ceo_name}}
Title: Chief Executive Officer
Date: _______________________
Signature: _______________________

**Chief Information Security Officer**

Name: {{ciso_name}}
Title: Chief Information Security Officer
Date: _______________________
Signature: _______________________

---

*This attestation is provided in connection with the SOC 2 Type II examination
for the period {{audit_period}}.*

**Document Hash:** {{document_hash}}
""",
    TemplateType.MANAGEMENT_ASSERTION_LETTER: """
# Management Assertion Letter

## {{company_name}}

**Document ID:** {{document_id}}
**Date:** {{current_date}}
**Report Period:** {{audit_period}}

---

To: {{auditor_firm_name}}

We, the management of {{company_name}} ("the Company"), provide this assertion
letter in connection with your examination of our system and the suitability
of the design and operating effectiveness of controls to meet the criteria for
the {{tsc_categories_text}} trust services categories.

### Management's Assertion

We assert that:

1. **System Description Accuracy**: The description of the {{system_name}} system
   as of {{report_date}} and throughout the period {{audit_period}} is fairly
   presented based on the criteria for such descriptions as set forth in
   DC Section 200, Description Criteria for a Description of a Service
   Organization's System in a SOC 2 Report.

2. **Control Suitability**: The controls stated in the description were suitably
   designed throughout the period {{audit_period}} to meet the applicable trust
   services criteria.

3. **Control Operating Effectiveness**: The controls stated in the description
   operated effectively throughout the period {{audit_period}} to meet the
   applicable trust services criteria.

### Basis for Assertion

This assertion is based on the criteria established by the American Institute
of Certified Public Accountants (AICPA) in TSP Section 100, 2017 Trust Services
Criteria for Security, Availability, Processing Integrity, Confidentiality,
and Privacy.

### Inherent Limitations

Because of their nature, controls may not prevent, or detect and correct, all
misstatements or omissions in processing or reporting transactions. Also, the
projection of any evaluation of effectiveness to future periods is subject to
the risk that controls may become inadequate because of changes in conditions,
or that the degree of compliance with the policies or procedures may deteriorate.

### Subservice Organizations

The following subservice organizations' controls are carved out of the system
description:

{{#subservice_organizations}}
- {{organization_name}}: {{service_description}}
{{/subservice_organizations}}

{{#no_subservice_organizations}}
No subservice organizations are carved out of the system description.
{{/no_subservice_organizations}}

---

## Signatures

**Chief Executive Officer**

Name: {{ceo_name}}
Title: Chief Executive Officer
Date: _______________________
Signature: _______________________

**Chief Financial Officer**

Name: {{cfo_name}}
Title: Chief Financial Officer
Date: _______________________
Signature: _______________________

---

**Document Hash:** {{document_hash}}
""",
    TemplateType.CONTROL_OWNER_ATTESTATION: """
# Control Owner Attestation

## {{company_name}}

**Document ID:** {{document_id}}
**Attestation Period:** {{attestation_period}}
**Control Area:** {{control_area}}

---

## Control Owner Attestation Statement

I, {{control_owner_name}}, as the designated control owner for {{control_area}}
at {{company_name}}, hereby attest to the following:

### 1. Control Operation

I confirm that the controls assigned to me have been operating as designed
throughout the attestation period {{attestation_period}}.

### 2. Controls Under My Responsibility

The following controls are under my ownership:

{{#controls}}
| Control ID | Control Description | Status |
|------------|---------------------|--------|
| {{control_id}} | {{control_description}} | {{control_status}} |
{{/controls}}

### 3. Evidence Collection

I confirm that:

- Evidence of control operation has been collected and retained.
- Evidence is accurate and represents actual control operation.
- Evidence is stored in accordance with the evidence retention policy.

### 4. Exceptions and Deviations

{{#has_exceptions}}
The following exceptions or deviations occurred during the attestation period:

{{#exceptions}}
- **{{exception_date}}**: {{exception_description}}
  - Root Cause: {{root_cause}}
  - Remediation: {{remediation_action}}
{{/exceptions}}
{{/has_exceptions}}

{{#no_exceptions}}
No exceptions or deviations occurred during the attestation period.
{{/no_exceptions}}

### 5. Changes to Controls

{{#has_changes}}
The following changes were made to controls during the attestation period:

{{#changes}}
- **{{change_date}}**: {{change_description}}
{{/changes}}
{{/has_changes}}

{{#no_changes}}
No changes were made to controls during the attestation period.
{{/no_changes}}

---

## Signature

**Control Owner**

Name: {{control_owner_name}}
Title: {{control_owner_title}}
Department: {{control_owner_department}}
Date: _______________________
Signature: _______________________

---

**Document Hash:** {{document_hash}}
""",
    TemplateType.SUBSERVICE_ORGANIZATION_LIST: """
# Subservice Organization List

## {{company_name}}

**Document ID:** {{document_id}}
**Effective Date:** {{effective_date}}
**Last Updated:** {{current_date}}

---

## Overview

This document identifies subservice organizations whose services are used by
{{company_name}} in providing {{service_description}}. For each subservice
organization, the document specifies whether the organization's controls are
included (inclusive method) or excluded (carve-out method) from our SOC 2
Type II examination.

---

## Subservice Organizations

{{#subservice_organizations}}
### {{organization_name}}

| Attribute | Value |
|-----------|-------|
| **Organization Name** | {{organization_name}} |
| **Service Description** | {{service_description}} |
| **Method** | {{method}} |
| **SOC Report Available** | {{soc_report_available}} |
| **Last SOC Report Date** | {{last_soc_report_date}} |
| **Report Type** | {{soc_report_type}} |
| **Monitoring Frequency** | {{monitoring_frequency}} |
| **Contract End Date** | {{contract_end_date}} |

**Services Used:**
{{service_details}}

**Complementary Subservice Organization Controls (CSOCs):**
{{#csoc_list}}
- {{csoc_description}}
{{/csoc_list}}

---

{{/subservice_organizations}}

## Summary

| Category | Count |
|----------|-------|
| Total Subservice Organizations | {{total_count}} |
| Carve-Out Method | {{carve_out_count}} |
| Inclusive Method | {{inclusive_count}} |
| With Current SOC Report | {{with_soc_report_count}} |

---

## Management Acknowledgment

Management acknowledges that:

1. All material subservice organizations have been identified in this document.
2. Appropriate due diligence has been performed on each subservice organization.
3. Monitoring and oversight activities are in place for each subservice organization.
4. Users are notified of any changes to subservice organizations that may affect them.

**Authorized By:**

Name: {{authorized_by_name}}
Title: {{authorized_by_title}}
Date: _______________________
Signature: _______________________

---

**Document Hash:** {{document_hash}}
""",
    TemplateType.COMPLEMENTARY_USER_ENTITY_CONTROLS: """
# Complementary User Entity Controls (CUECs)

## {{company_name}}

**Document ID:** {{document_id}}
**Effective Date:** {{effective_date}}
**Version:** {{version}}

---

## Introduction

This document describes the Complementary User Entity Controls (CUECs) that
user entities (customers) of {{company_name}}'s {{service_name}} are expected
to implement. These controls complement the controls at {{company_name}} and
are necessary for the overall control environment to be effective.

The controls described in this document are assumed to be in place at user
entities in conjunction with controls at {{company_name}} as part of the
SOC 2 examination for the {{tsc_categories_text}} trust services categories.

---

## User Entity Responsibilities

### 1. Access Management

{{#access_controls}}
| CUEC ID | Control Description | Criteria |
|---------|---------------------|----------|
| {{cuec_id}} | {{description}} | {{criteria}} |
{{/access_controls}}

### 2. Data Protection

{{#data_controls}}
| CUEC ID | Control Description | Criteria |
|---------|---------------------|----------|
| {{cuec_id}} | {{description}} | {{criteria}} |
{{/data_controls}}

### 3. System Configuration

{{#system_controls}}
| CUEC ID | Control Description | Criteria |
|---------|---------------------|----------|
| {{cuec_id}} | {{description}} | {{criteria}} |
{{/system_controls}}

### 4. Monitoring and Response

{{#monitoring_controls}}
| CUEC ID | Control Description | Criteria |
|---------|---------------------|----------|
| {{cuec_id}} | {{description}} | {{criteria}} |
{{/monitoring_controls}}

### 5. Business Continuity

{{#continuity_controls}}
| CUEC ID | Control Description | Criteria |
|---------|---------------------|----------|
| {{cuec_id}} | {{description}} | {{criteria}} |
{{/continuity_controls}}

---

## Summary

| Category | Number of CUECs |
|----------|-----------------|
| Access Management | {{access_count}} |
| Data Protection | {{data_count}} |
| System Configuration | {{system_count}} |
| Monitoring and Response | {{monitoring_count}} |
| Business Continuity | {{continuity_count}} |
| **Total** | **{{total_cuec_count}}** |

---

## Communication

User entities are notified of CUECs through:

1. Inclusion in service agreements and contracts
2. Publication in customer documentation portal
3. Annual review and acknowledgment process
4. Communication of changes via email notification

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
{{#version_history}}
| {{version}} | {{date}} | {{changes}} | {{author}} |
{{/version_history}}

---

**Document Hash:** {{document_hash}}
""",
}


# ---------------------------------------------------------------------------
# Attestation Templates
# ---------------------------------------------------------------------------


class AttestationTemplates:
    """Template generation and population for attestation documents.

    Provides methods to generate documents from templates, substitute
    variables, and export to PDF format.

    Attributes:
        TEMPLATES: Dictionary mapping template types to their content.

    Example:
        >>> templates = AttestationTemplates()
        >>> variables = {
        ...     "company_name": "GreenLang Inc.",
        ...     "audit_period": "January 1, 2026 - December 31, 2026",
        ... }
        >>> doc = templates.generate_document("soc2_readiness_attestation", variables)
    """

    TEMPLATES = _TEMPLATES

    def __init__(self) -> None:
        """Initialize AttestationTemplates."""
        logger.info("AttestationTemplates initialized with %d templates", len(self.TEMPLATES))

    def generate_document(
        self,
        template_type: str,
        variables: Dict[str, Any],
        generated_by: str = "",
    ) -> Document:
        """Generate a document from a template with variable substitution.

        Args:
            template_type: Type of template to use.
            variables: Dictionary of variables to substitute in the template.
            generated_by: User ID of the person generating the document.

        Returns:
            Generated Document with populated content.

        Raises:
            ValueError: If template_type is invalid.
        """
        start_time = datetime.now(timezone.utc)

        # Validate template type
        try:
            tpl_type = TemplateType(template_type)
        except ValueError as e:
            valid_types = [t.value for t in TemplateType]
            raise ValueError(
                f"Invalid template_type '{template_type}'. "
                f"Valid types: {valid_types}"
            ) from e

        # Get template content
        template_content = self.TEMPLATES.get(tpl_type)
        if template_content is None:
            raise ValueError(f"Template '{template_type}' not found.")

        # Generate document ID and add standard variables
        document_id = str(uuid.uuid4())
        standard_vars = {
            "document_id": document_id,
            "current_date": datetime.now(timezone.utc).strftime("%B %d, %Y"),
            "document_hash": "",  # Placeholder, updated after content generation
        }
        all_variables = {**standard_vars, **variables}

        # Populate template
        content = self.populate_variables(template_content, all_variables)

        # Calculate content hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Update document hash in content
        content = content.replace("{{document_hash}}", content_hash)

        # Determine title based on template type
        title_map = {
            TemplateType.SOC2_READINESS_ATTESTATION: "SOC 2 Type II Readiness Attestation",
            TemplateType.MANAGEMENT_ASSERTION_LETTER: "Management Assertion Letter",
            TemplateType.CONTROL_OWNER_ATTESTATION: "Control Owner Attestation",
            TemplateType.SUBSERVICE_ORGANIZATION_LIST: "Subservice Organization List",
            TemplateType.COMPLEMENTARY_USER_ENTITY_CONTROLS: "Complementary User Entity Controls",
        }
        title = title_map.get(tpl_type, "Attestation Document")

        # Create document
        document = Document(
            document_id=document_id,
            template_type=tpl_type,
            title=title,
            content=content,
            content_hash=content_hash,
            variables_used=variables,
            generated_by=generated_by,
        )

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Document generated: id=%s, type=%s, title='%s', "
            "variables=%d, length=%d, elapsed=%.2fms",
            document_id,
            tpl_type.value,
            title,
            len(variables),
            len(content),
            elapsed_ms,
        )

        return document

    def populate_variables(
        self,
        template: str,
        variables: Dict[str, Any],
    ) -> str:
        """Substitute variables in a template string.

        Supports simple {{variable}} substitution. For complex templates
        with conditionals and loops, consider using a full templating engine.

        Args:
            template: Template string with {{variable}} placeholders.
            variables: Dictionary of variable names to values.

        Returns:
            Template with variables substituted.
        """
        result = template

        # Simple variable substitution
        for key, value in variables.items():
            placeholder = "{{" + key + "}}"
            if placeholder in result:
                # Convert value to string if necessary
                str_value = str(value) if value is not None else ""
                result = result.replace(placeholder, str_value)

        # Log any remaining placeholders
        remaining = re.findall(r"\{\{[^}]+\}\}", result)
        if remaining:
            logger.debug(
                "Unpopulated placeholders in template: %s",
                remaining[:10],  # Limit to first 10
            )

        return result

    def export_pdf(self, document: Document) -> bytes:
        """Export a document to PDF format.

        Note: This is a placeholder implementation. In production, integrate
        with a PDF generation library (e.g., WeasyPrint, ReportLab, or
        an external service).

        Args:
            document: Document to export.

        Returns:
            PDF file contents as bytes.
        """
        logger.info(
            "Exporting document to PDF: id=%s, title='%s'",
            document.document_id,
            document.title,
        )

        # Placeholder: Return a simple PDF-like header with content
        # In production, use a proper PDF library
        pdf_header = b"%PDF-1.4\n"
        content_bytes = document.content.encode("utf-8")

        # Create a minimal PDF structure (not a real PDF, just a placeholder)
        # Real implementation would use ReportLab, WeasyPrint, or similar
        placeholder_pdf = (
            pdf_header
            + b"% Placeholder PDF - integrate with PDF library for production\n"
            + b"% Document ID: "
            + document.document_id.encode("utf-8")
            + b"\n"
            + b"% Title: "
            + document.title.encode("utf-8")
            + b"\n"
            + b"% Content Hash: "
            + document.content_hash.encode("utf-8")
            + b"\n"
            + b"% Content Length: "
            + str(len(content_bytes)).encode("utf-8")
            + b"\n"
            + b"%%EOF\n"
        )

        logger.debug(
            "PDF exported: id=%s, size=%d bytes",
            document.document_id,
            len(placeholder_pdf),
        )

        return placeholder_pdf

    def get_template(self, template_type: str) -> Optional[str]:
        """Get the raw template content for a template type.

        Args:
            template_type: Type of template to retrieve.

        Returns:
            Template content string, or None if not found.
        """
        try:
            tpl_type = TemplateType(template_type)
            return self.TEMPLATES.get(tpl_type)
        except ValueError:
            return None

    def list_templates(self) -> Dict[str, str]:
        """List all available templates with their descriptions.

        Returns:
            Dictionary mapping template type values to descriptions.
        """
        descriptions = {
            TemplateType.SOC2_READINESS_ATTESTATION: (
                "CEO/CISO sign pre-audit to attest readiness for SOC 2 examination"
            ),
            TemplateType.MANAGEMENT_ASSERTION_LETTER: (
                "CEO/CFO sign with report to assert control effectiveness"
            ),
            TemplateType.CONTROL_OWNER_ATTESTATION: (
                "Control owners sign quarterly to confirm control operation"
            ),
            TemplateType.SUBSERVICE_ORGANIZATION_LIST: (
                "List of carved-out services with management acknowledgment"
            ),
            TemplateType.COMPLEMENTARY_USER_ENTITY_CONTROLS: (
                "Customer responsibilities that complement service organization controls"
            ),
        }
        return {t.value: descriptions.get(t, "") for t in TemplateType}

    def get_required_variables(self, template_type: str) -> list[str]:
        """Get the list of variables used in a template.

        Extracts all {{variable}} placeholders from the template.

        Args:
            template_type: Type of template to analyze.

        Returns:
            List of variable names found in the template.
        """
        template = self.get_template(template_type)
        if template is None:
            return []

        # Extract all {{variable}} placeholders
        matches = re.findall(r"\{\{([^}#/]+)\}\}", template)

        # Deduplicate while preserving order
        seen: set[str] = set()
        variables: list[str] = []
        for match in matches:
            var_name = match.strip()
            if var_name not in seen:
                seen.add(var_name)
                variables.append(var_name)

        return variables


__all__ = [
    "TemplateType",
    "Document",
    "AttestationTemplates",
]
