"""
PACK-026: SME Net Zero Pack
============================

Comprehensive GreenLang deployment pack optimized for Small and Medium Enterprises
(SMEs) with <250 employees. Provides simplified, cost-effective net zero solutions
with minimal data requirements and maximum ROI.

Key SME Features:
-----------------
- **Three-tier data approach**: Bronze (±40%, 15 min), Silver (±15%, 1 hour), Gold (±5%, 2-3 hours)
- **Quick wins database**: 50+ actions with <3 year payback
- **Grant finder**: 50+ UK/EU/US programs with automatic matching
- **Accounting integration**: Xero, QuickBooks, Sage auto-import
- **Simplified targets**: 1.5°C pathway, 50% by 2030 (pre-configured)
- **Certification pathways**: SME Climate Hub, B Corp, ISO 14001, Carbon Trust
- **Peer benchmarking**: Compare to similar-sized SMEs
- **Express onboarding**: <15 minutes to baseline and targets

Components:
-----------
- **8 Engines**: SME-optimized calculations with minimal data requirements
- **6 Workflows**: Streamlined for speed (<15 min express, <2 hours full setup)
- **8 Templates**: 1-2 page reports, mobile-optimized, visual focus
- **13 Integrations**: Accounting software, grants, certification bodies
- **6 Presets**: Micro/small/medium businesses across service/manufacturing/retail

Market:
-------
- 99% of businesses globally are SMEs
- Collectively responsible for 50%+ of global emissions
- Target implementation cost: <€5,000 (micro), <€20,000 (medium)
- Target ROI: >3:1 from grant funding + energy savings

Standards:
----------
- GHG Protocol Corporate Standard (simplified)
- SBTi SME Pathway (2023)
- SME Climate Hub (UN-backed)
- B Corp Climate Collective
- ISO 14001 Environmental Management System
- Carbon Trust Standard
- DEFRA/EPA EEIO Emission Factors

Version: 1.0.0
Release: 2026-03-18
Author: GreenLang Platform Team
License: Proprietary

Example Usage:
--------------
```python
from pack026_sme_net_zero import (
    SMEBaselineEngine,
    SimplifiedTargetEngine,
    QuickWinsEngine,
    GrantFinderEngine,
    ExpressOnboardingWorkflow,
    SMEBaselineReport,
    XeroConnector,
    sme_preset_loader,
)

# 1. Express onboarding (15 minutes)
preset = sme_preset_loader.load_preset("small_business")
workflow = ExpressOnboardingWorkflow(config=preset)
result = await workflow.execute(
    company_name="Example SME Ltd",
    industry_nace="62.01",  # Computer programming
    employees=25,
    revenue_eur=2_500_000,
    annual_energy_spend_eur=50_000,
)

# 2. Generate baseline report
report = SMEBaselineReport()
baseline_pdf = await report.render_pdf(result.baseline)

# 3. Find quick wins
quick_wins = QuickWinsEngine()
wins = await quick_wins.identify(
    baseline=result.baseline,
    sector="tech_services",
    budget_max_eur=10_000,
)

# 4. Find matching grants
grant_finder = GrantFinderEngine()
grants = await grant_finder.match(
    industry_code="62.01",
    size_tier="SMALL",
    country="UK",
    postcode="EC2A",
    emissions_profile=result.baseline,
)

# 5. Connect accounting software
xero = XeroConnector()
await xero.authenticate(oauth_token="...")
spend_data = await xero.import_spend(
    from_date="2024-01-01",
    to_date="2024-12-31",
)
```

For detailed documentation, see: https://docs.greenlang.io/packs/sme-net-zero
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-026-sme-net-zero"
__pack_name__ = "SME Net Zero Pack"
__category__: str = "net-zero"

# Re-export main components for convenient imports
from . import engines
from . import workflows
from . import templates
from . import integrations
from . import config

# Engines
from .engines import (
    SMEBaselineEngine,
    SimplifiedTargetEngine,
    QuickWinsEngine,
    Scope3EstimatorEngine,
    ActionPrioritizationEngine,
    CostBenefitEngine,
    GrantFinderEngine,
    CertificationReadinessEngine,
)

# Workflows
from .workflows import (
    ExpressOnboardingWorkflow,
    StandardSetupWorkflow,
    GrantApplicationWorkflow,
    QuarterlyReviewWorkflow,
    QuickWinsImplementationWorkflow,
    CertificationPathwayWorkflow,
)

# Templates
from .templates import (
    SMEBaselineReport,
    SMEQuickWinsReport,
    SMEGrantReport,
    SMEBoardBrief,
    SMERoadmapReport,
    SMEProgressDashboard,
    SMECertificationSubmission,
    SMEAccountingGuide,
)

# Integrations
from .integrations import (
    PackOrchestrator,
    MRVBridge,
    DataBridge,
    XeroConnector,
    QuickBooksConnector,
    SageConnector,
    GrantDatabaseBridge,
    SMEClimateHubBridge,
    CertificationBodyBridge,
    PeerNetworkBridge,
    RenewablePPAMarketplace,
    SetupWizard,
    HealthCheck,
)

# Configuration
from .config import (
    SMEPackConfig,
    load_preset,
    load_preset_for_sector,
    list_presets,
)

__all__ = [
    # Metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",

    # Modules
    "engines",
    "workflows",
    "templates",
    "integrations",
    "config",

    # Engines
    "SMEBaselineEngine",
    "SimplifiedTargetEngine",
    "QuickWinsEngine",
    "Scope3EstimatorEngine",
    "ActionPrioritizationEngine",
    "CostBenefitEngine",
    "GrantFinderEngine",
    "CertificationReadinessEngine",

    # Workflows
    "ExpressOnboardingWorkflow",
    "StandardSetupWorkflow",
    "GrantApplicationWorkflow",
    "QuarterlyReviewWorkflow",
    "QuickWinsImplementationWorkflow",
    "CertificationPathwayWorkflow",

    # Templates
    "SMEBaselineReport",
    "SMEQuickWinsReport",
    "SMEGrantReport",
    "SMEBoardBrief",
    "SMERoadmapReport",
    "SMEProgressDashboard",
    "SMECertificationSubmission",
    "SMEAccountingGuide",

    # Integrations
    "PackOrchestrator",
    "MRVBridge",
    "DataBridge",
    "XeroConnector",
    "QuickBooksConnector",
    "SageConnector",
    "GrantDatabaseBridge",
    "SMEClimateHubBridge",
    "CertificationBodyBridge",
    "PeerNetworkBridge",
    "RenewablePPAMarketplace",
    "SetupWizard",
    "HealthCheck",

    # Configuration
    "SMEPackConfig",
    "load_preset",
    "load_preset_for_sector",
    "list_presets",
]
