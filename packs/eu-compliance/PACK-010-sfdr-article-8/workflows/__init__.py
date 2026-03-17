# -*- coding: utf-8 -*-
"""
PACK-010 SFDR Article 8 Pack - Workflow Orchestration
========================================================

Production-grade workflow orchestrators for SFDR Article 8 financial
product compliance operations. Each workflow coordinates disclosure
generation, portfolio screening, PAI calculation, taxonomy alignment,
and regulatory change management into end-to-end SFDR processes aligned
with EU Regulation 2019/2088 (SFDR) and Delegated Regulation 2022/1288.

Workflows:
    - PrecontractualDisclosureWorkflow: 5-phase Annex II disclosure
      generation covering product classification, investment strategy,
      sustainability assessment, template population, and review/approval.

    - PeriodicReportingWorkflow: 5-phase Annex IV periodic disclosure
      with data collection, performance assessment, PAI calculation,
      template generation, and filing package assembly.

    - WebsiteDisclosureWorkflow: 4-phase Annex III website disclosure
      with content assembly, multi-format template generation, version
      tracking, and accessibility-compliant publication.

    - PAIStatementWorkflow: 4-phase PAI calculation and statement
      generation with investee-level data sourcing, portfolio-weighted
      calculations for all 18 mandatory indicators, reporting, and
      action planning for worst performers.

    - PortfolioScreeningWorkflow: 4-phase investment screening with
      universe definition, negative screening (5 default exclusion
      rules), positive screening with weighted scoring, and compliance
      check against binding elements.

    - TaxonomyAlignmentWorkflow: 4-phase EU Taxonomy alignment ratio
      calculation with holdings analysis, DNSH/safeguards assessment,
      portfolio aggregation with double-counting prevention, and
      commitment tracking.

    - ComplianceReviewWorkflow: 4-phase disclosure completeness and
      compliance review with disclosure freshness checking, PAI data
      quality assessment, binding element adherence verification, and
      prioritized action item generation.

    - RegulatoryUpdateWorkflow: 3-phase regulatory change management
      with change detection (SFDR amendments, ESMA guidance, RTS
      changes, SFDR 2.0), impact assessment, and migration planning
      with implementation roadmap.

Shared Infrastructure:
    - WorkflowStatus / PhaseStatus enums for consistent state tracking
    - WorkflowContext for inter-phase state propagation
    - PhaseResult / WorkflowResult Pydantic models with provenance hashes
    - Checkpoint/resume support via phase-level status tracking
    - Phase skip support via config.skip_phases

Author: GreenLang Team
Version: 1.0.0
"""

from packs.eu_compliance.PACK_010_sfdr_article_8.workflows.precontractual_disclosure import (
    PrecontractualDisclosureWorkflow,
    PrecontractualDisclosureInput,
    PrecontractualDisclosureResult,
)
from packs.eu_compliance.PACK_010_sfdr_article_8.workflows.periodic_reporting import (
    PeriodicReportingWorkflow,
    PeriodicReportingInput,
    PeriodicReportingResult,
)
from packs.eu_compliance.PACK_010_sfdr_article_8.workflows.website_disclosure import (
    WebsiteDisclosureWorkflow,
    WebsiteDisclosureInput,
    WebsiteDisclosureResult,
)
from packs.eu_compliance.PACK_010_sfdr_article_8.workflows.pai_statement import (
    PAIStatementWorkflow,
    PAIStatementInput,
    PAIStatementResult,
)
from packs.eu_compliance.PACK_010_sfdr_article_8.workflows.portfolio_screening import (
    PortfolioScreeningWorkflow,
    PortfolioScreeningInput,
    PortfolioScreeningResult,
)
from packs.eu_compliance.PACK_010_sfdr_article_8.workflows.taxonomy_alignment import (
    TaxonomyAlignmentWorkflow,
    TaxonomyAlignmentInput,
    TaxonomyAlignmentResult,
)
from packs.eu_compliance.PACK_010_sfdr_article_8.workflows.compliance_review import (
    ComplianceReviewWorkflow,
    ComplianceReviewInput,
    ComplianceReviewResult,
)
from packs.eu_compliance.PACK_010_sfdr_article_8.workflows.regulatory_update import (
    RegulatoryUpdateWorkflow,
    RegulatoryUpdateInput,
    RegulatoryUpdateResult,
)

__version__ = "1.0.0"
__author__ = "GreenLang Team"

__all__ = [
    # Pre-contractual Disclosure (Annex II)
    "PrecontractualDisclosureWorkflow",
    "PrecontractualDisclosureInput",
    "PrecontractualDisclosureResult",
    # Periodic Reporting (Annex IV)
    "PeriodicReportingWorkflow",
    "PeriodicReportingInput",
    "PeriodicReportingResult",
    # Website Disclosure (Annex III)
    "WebsiteDisclosureWorkflow",
    "WebsiteDisclosureInput",
    "WebsiteDisclosureResult",
    # PAI Statement
    "PAIStatementWorkflow",
    "PAIStatementInput",
    "PAIStatementResult",
    # Portfolio Screening
    "PortfolioScreeningWorkflow",
    "PortfolioScreeningInput",
    "PortfolioScreeningResult",
    # Taxonomy Alignment
    "TaxonomyAlignmentWorkflow",
    "TaxonomyAlignmentInput",
    "TaxonomyAlignmentResult",
    # Compliance Review
    "ComplianceReviewWorkflow",
    "ComplianceReviewInput",
    "ComplianceReviewResult",
    # Regulatory Update
    "RegulatoryUpdateWorkflow",
    "RegulatoryUpdateInput",
    "RegulatoryUpdateResult",
]
