# -*- coding: utf-8 -*-
"""
PACK-050: GHG Consolidation Pack
==================================

GreenLang deployment pack providing comprehensive multi-entity corporate
GHG consolidation capabilities per GHG Protocol Corporate Standard
Chapter 3. The pack covers entity registry and corporate structure
management, multi-tier ownership chain resolution, organisational
boundary determination under equity share / operational control /
financial control consolidation approaches, inter-company emission
elimination, M&A event handling with pro-rata apportionment, adjustment
and restatement processing, group reporting with multi-framework
regulatory output, and comprehensive audit trail generation.

This pack complements the calculation packs (PACK-041 Scope 1-2 Complete,
PACK-042 Scope 3 Starter, PACK-043 Scope 3 Complete), the governance pack
(PACK-044 Inventory Management), the base year pack (PACK-045 Base Year
Management), the intensity metrics pack (PACK-046 Intensity Metrics), the
benchmark pack (PACK-047 GHG Emissions Benchmark), the assurance prep
pack (PACK-048 GHG Assurance Prep), and the multi-site management pack
(PACK-049 Multi-Site Management) by providing the dedicated corporate
consolidation layer that aggregates entity-level GHG inventories into a
single consolidated corporate inventory per the chosen organisational
boundary approach.

Core Capabilities:
    1. Entity Registry - Corporate structure, classification, lifecycle management
    2. Ownership Chain - Multi-tier ownership resolution, effective equity calculation
    3. Boundary Determination - Org boundary per consolidation approach with materiality
    4. Equity Share Consolidation - Proportional emissions per equity stake
    5. Control Approach Consolidation - 100%/0% inclusion per control determination
    6. Inter-Company Elimination - Netting of intra-group energy/waste/product/service transfers
    7. M&A Event Handling - Acquisition/divestiture/merger/demerger with pro-rata
    8. Adjustment Processing - Methodology changes, error corrections, restatements
    9. Group Reporting - Consolidated reports with multi-framework regulatory output
    10. Audit Trail - Reconciliation, variance analysis, sign-off, evidence packaging

Engines (10):
    1. EntityRegistryEngine           - Entity registry and corporate structure
    2. OwnershipChainEngine          - Multi-tier ownership resolution
    3. BoundaryDeterminationEngine   - Organisational boundary per approach
    4. EquityShareEngine             - Equity share proportional consolidation
    5. ControlApproachEngine         - Operational/financial control consolidation
    6. EliminationEngine             - Inter-company emission elimination
    7. MnAEventEngine                - M&A event processing with pro-rata
    8. AdjustmentEngine              - Adjustment and restatement processing
    9. GroupReportingEngine          - Consolidated group report generation
    10. ConsolidationAuditEngine     - Audit trail and reconciliation

Workflows (8):
    1. EntityOnboardingWorkflow         - New entity registration and setup
    2. BoundaryReviewWorkflow           - Annual boundary review and update
    3. ConsolidationRunWorkflow         - Period-end consolidation execution
    4. MnAProcessingWorkflow            - M&A event capture and processing
    5. AdjustmentWorkflow               - Adjustment request and application
    6. GroupReportingWorkflow            - Multi-framework report generation
    7. AuditPreparationWorkflow         - Audit evidence and package preparation
    8. FullConsolidationPipelineWorkflow - End-to-end consolidation orchestration

Regulatory Basis:
    GHG Protocol Corporate Standard (2004, revised 2015) - Chapter 3
    GHG Protocol Scope 2 Guidance (2015) - Dual reporting
    ISO 14064-1:2018 Clause 5 - Organisational boundaries
    EU CSRD (2022/2464) - ESRS E1 consolidated disclosure
    US SEC Climate Disclosure Rules (2024) - Registrant boundary
    IFRS S2 - Climate-related financial disclosures
    GRI 305 (2016) - Consolidated GHG emissions
    CDP Climate Change - C6/C7 consolidation
    SBTi Corporate Net-Zero Standard - Boundary requirements
    PCAF Global Standard v3 (2024) - Financed emissions consolidation

Category: GHG Accounting Packs
Pack Tier: Enterprise
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

__version__: str = "1.0.0"
__pack__: str = "PACK-050"
__pack_name__: str = "GHG Consolidation Pack"
__category__: str = "ghg-accounting"
