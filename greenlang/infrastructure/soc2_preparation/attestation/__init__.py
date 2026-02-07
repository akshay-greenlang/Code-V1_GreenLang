# -*- coding: utf-8 -*-
"""
Management Attestation Module - SEC-009 Phase 7

This module provides management attestation workflows for SOC 2 Type II audit preparation.
Supports digital signature collection, template-based document generation, and
complete audit trail tracking for regulatory compliance.

Submodules:
    - workflow: Attestation state machine and lifecycle management
    - templates: Pre-built attestation document templates
    - signer: Digital signature integration (DocuSign, Adobe Sign, internal)

Public API:
    - AttestationWorkflow: Main workflow orchestration class
    - AttestationTemplates: Template generation and population
    - DigitalSigner: Signature collection and verification
    - AttestationStatus: Attestation lifecycle states
    - SignatureMethod: Supported signature providers

Example:
    >>> from greenlang.infrastructure.soc2_preparation.attestation import (
    ...     AttestationWorkflow,
    ...     AttestationTemplates,
    ...     DigitalSigner,
    ... )
    >>> workflow = AttestationWorkflow(config)
    >>> attestation = await workflow.create_attestation(
    ...     attestation_type="soc2_readiness_attestation",
    ...     document_name="Q4 2026 SOC 2 Readiness",
    ... )
    >>> await workflow.submit_for_review(attestation.attestation_id)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging

from greenlang.infrastructure.soc2_preparation.attestation.workflow import (
    AttestationWorkflow,
    AttestationStatus,
    Attestation,
    AttestationCreate,
)
from greenlang.infrastructure.soc2_preparation.attestation.templates import (
    AttestationTemplates,
    Document,
    TemplateType,
)
from greenlang.infrastructure.soc2_preparation.attestation.signer import (
    DigitalSigner,
    SignatureMethod,
    SignatureStatus,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Workflow
    "AttestationWorkflow",
    "AttestationStatus",
    "Attestation",
    "AttestationCreate",
    # Templates
    "AttestationTemplates",
    "Document",
    "TemplateType",
    # Signer
    "DigitalSigner",
    "SignatureMethod",
    "SignatureStatus",
]
