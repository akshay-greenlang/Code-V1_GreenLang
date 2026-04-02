# -*- coding: utf-8 -*-
"""
Shared Supplier Data Types
==========================

Cross-cutting enumerations and models used by both the Supplier Data
Exchange Agent (GL-DATA-X-012) and the Supplier Questionnaire Processor
(AGENT-DATA-008).

Extracted types:
    - DataQualityRating: Quality tier classification for supplier-provided data
    - SupplierValidationOutcome: Generic validation outcome (pass/fail/warning)
    - SubmissionStatus: Lifecycle status for supplier data submissions
    - SupplierIdentity: Base model for supplier identification across agents

These types capture supplier data quality, validation, identification, and
submission lifecycle concepts that apply across both PCF data exchange and
questionnaire processing workflows.

Author: GreenLang Team
Version: 1.1.0
"""

from enum import Enum
from typing import Optional

from pydantic import Field

from greenlang.schemas import GreenLangBase


class DataQualityRating(str, Enum):
    """Data quality rating for supplier-provided information.

    Classifies the provenance and reliability tier of data submitted
    by suppliers, whether through PCF data exchange or questionnaire
    responses. Aligns with GHG Protocol data quality hierarchy.

    PRIMARY: Supplier-specific measured data (highest quality).
    SECONDARY: Industry average or proxy data from published sources.
    TERTIARY: Default, estimated, or modelled data (lowest quality).
    UNKNOWN: Data quality tier has not been assessed or is unavailable.
    """

    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    UNKNOWN = "unknown"


class SupplierValidationOutcome(str, Enum):
    """Generic validation outcome for supplier data checks.

    Represents the result of a single validation check applied to
    supplier-submitted data. Used by both PCF submission validation
    and questionnaire response validation pipelines.

    PASS: The check passed successfully; data meets the requirement.
    FAIL: The check failed; data does not meet the requirement.
    WARNING: The check raised a non-blocking concern; review recommended.
    """

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class SubmissionStatus(str, Enum):
    """Lifecycle status for supplier data submissions.

    Tracks the lifecycle of a supplier data submission (PCF data,
    questionnaire response, etc.) from initial receipt through
    validation to final acceptance or rejection.

    PENDING: Submission received, awaiting validation.
    VALIDATED: Submission passed all validation checks.
    REJECTED: Submission failed validation and was rejected.
    NEEDS_REVISION: Submission requires corrections from the supplier.
    ACCEPTED: Submission accepted and integrated into the system.
    EXPIRED: Submission expired (e.g. superseded or past deadline).
    """

    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    ACCEPTED = "accepted"
    EXPIRED = "expired"


class SupplierIdentity(GreenLangBase):
    """Base model for supplier identification.

    Captures the core identifying attributes shared across supplier-related
    agents: PCF data exchange, questionnaire processing, and supply chain
    mapping. Downstream models extend this with domain-specific fields.

    Attributes:
        supplier_id: Unique supplier identifier.
        supplier_name: Supplier display name.
        contact_email: Primary contact email address.
        industry_sector: Industry classification or sector.
    """

    supplier_id: str = Field(..., description="Unique supplier identifier")
    supplier_name: str = Field(..., description="Supplier display name")
    contact_email: Optional[str] = Field(
        None, description="Primary contact email address"
    )
    industry_sector: Optional[str] = Field(
        None, description="Industry classification or sector"
    )


__all__ = [
    "DataQualityRating",
    "SubmissionStatus",
    "SupplierIdentity",
    "SupplierValidationOutcome",
]
