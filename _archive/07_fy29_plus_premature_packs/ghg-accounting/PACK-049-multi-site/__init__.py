# -*- coding: utf-8 -*-
"""
PACK-049: GHG Multi-Site Management Pack
==========================================

GreenLang deployment pack providing comprehensive multi-site GHG inventory
management capabilities including facility registry, decentralised data
collection, organisational boundary definition, regional emission factor
assignment, site-level consolidation, shared-services allocation, internal
benchmarking across facilities, completeness tracking, quality scoring,
and multi-site reporting.

This pack complements the calculation packs (PACK-041 Scope 1-2 Complete,
PACK-042 Scope 3 Starter, PACK-043 Scope 3 Complete), the governance pack
(PACK-044 Inventory Management), the base year pack (PACK-045 Base Year
Management), the intensity metrics pack (PACK-046 Intensity Metrics), the
benchmark pack (PACK-047 GHG Emissions Benchmark), and the assurance prep
pack (PACK-048 GHG Assurance Prep) by providing the dedicated multi-site
orchestration layer that registers and classifies facilities across an
organisational portfolio, enables decentralised data collection with
deadline management and submission tracking, defines organisational
boundaries under equity share, operational control, and financial control
consolidation approaches, assigns region-appropriate emission factors from
tiered sources (facility-specific, national, regional, IPCC default),
consolidates site-level inventories into group totals with equity
adjustment and elimination, allocates shared-services and landlord-tenant
emissions across tenants and cost centres, benchmarks facilities using
normalised KPIs (emissions per m2, per FTE, per revenue, per unit),
tracks data completeness and submission coverage across the portfolio,
scores data quality across six dimensions (accuracy, completeness,
consistency, timeliness, methodology, documentation), and generates
multi-site portfolio reports with drill-down capability.

Core Capabilities:
    1. Site Registry - Facility classification, lifecycle tracking, grouping
    2. Site Data Collection - Decentralised collection with deadlines and reminders
    3. Site Boundary - Organisational boundary with consolidation approach mapping
    4. Regional Factor Assignment - Tiered emission factor selection by geography
    5. Site Consolidation - Equity share / operational / financial control rollup
    6. Site Allocation - Shared services, landlord-tenant, cogeneration splits
    7. Site Comparison - Normalised KPI benchmarking across facilities
    8. Site Completion - Portfolio completeness tracking with gap identification
    9. Site Quality - Six-dimension data quality scoring with improvement tracking
    10. Multi-Site Reporting - Portfolio dashboards with site-level drill-down

Engines (10):
    1. SiteRegistryEngine         - Facility registry and classification management
    2. SiteDataCollectionEngine   - Decentralised data collection orchestration
    3. SiteBoundaryEngine         - Organisational boundary and scope definition
    4. RegionalFactorEngine       - Region-appropriate emission factor assignment
    5. SiteConsolidationEngine    - Multi-site inventory consolidation and rollup
    6. SiteAllocationEngine       - Shared services and tenant emission allocation
    7. SiteComparisonEngine       - Cross-facility KPI benchmarking
    8. SiteCompletionEngine       - Portfolio completeness and gap tracking
    9. SiteQualityEngine          - Multi-dimension data quality scoring
    10. MultiSiteReportingEngine  - Portfolio-level report generation

Workflows (8):
    1. SiteOnboardingWorkflow       - New facility registration and setup
    2. DataCollectionCycleWorkflow  - Period-end data collection orchestration
    3. BoundaryReviewWorkflow       - Annual boundary review and update
    4. FactorUpdateWorkflow         - Emission factor refresh and assignment
    5. ConsolidationWorkflow        - Group-level inventory consolidation
    6. AllocationWorkflow           - Shared services allocation execution
    7. BenchmarkingWorkflow         - Cross-site performance benchmarking
    8. FullMultiSitePipelineWorkflow - End-to-end multi-site orchestration

Regulatory Basis:
    GHG Protocol Corporate Standard (2004, revised 2015) - Chapter 3 & 4
    GHG Protocol Corporate Value Chain (Scope 3) Standard - Chapter 3
    ISO 14064-1:2018 Clause 5 - Organisational boundaries
    EU CSRD (2022/2464) - ESRS E1 multi-site disclosure requirements
    US SEC Climate Disclosure Rules (2024) - Registrant boundary guidance
    PCAF Global GHG Accounting Standard v3 (2024) - Portfolio aggregation

Category: GHG Accounting Packs
Pack Tier: Enterprise
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

__version__: str = "1.0.0"
__pack__: str = "PACK-049"
__pack_name__: str = "GHG Multi-Site Management Pack"
__category__: str = "ghg-accounting"
