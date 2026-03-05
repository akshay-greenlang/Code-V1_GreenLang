"""
GL-Taxonomy-APP v1.0 -- EU Taxonomy Alignment & Green Investment Ratio Platform Services

This package provides configuration, domain models, and 10 service engines plus
setup facade for implementing the EU Taxonomy Regulation 2020/852 alignment
assessment, KPI calculation, GAR/BTAR computation, and regulatory reporting
framework covering the Climate Delegated Act 2021/2139, Environmental Delegated
Act 2023/2486, Article 8 Delegated Act 2021/4987, and EBA Pillar 3 ITS.

Engines (10 service engines + config + models + setup):
    config: Environmental objectives, sectors, thresholds, and app settings.
    models: Pydantic v2 domain models for all EU Taxonomy entities.
    activity_screening_engine: NACE mapping, eligibility, sector classification.
    substantial_contribution_engine: TSC threshold evaluation, enabling/transitional.
    dnsh_assessment_engine: 6-objective DNSH matrix, climate risk assessment.
    minimum_safeguards_engine: 4-topic company-level safeguards assessment.
    kpi_calculation_engine: Turnover/CapEx/OpEx, double-counting prevention.
    gar_calculation_engine: GAR stock/flow, BTAR, EBA template computation.
    alignment_engine: End-to-end alignment workflow orchestration.
    reporting_engine: Article 8, EBA Pillar 3, XBRL disclosure reporting.
    data_quality_engine: 5-dimension quality scoring and assessment.
    regulatory_update_engine: DA version tracking, Omnibus simplification.
    setup: Platform facade composing all engines with FastAPI app factory.

Standard: EU Taxonomy Regulation 2020/852
Climate DA: Commission Delegated Regulation 2021/2139
Environmental DA: Commission Delegated Regulation 2023/2486
Article 8 DA: Commission Delegated Regulation 2021/4987
EBA ITS: EBA ITS on Pillar 3 ESG Disclosures (CRR Article 449a)
Environmental Objectives: 6 (climate mitigation, climate adaptation,
    water, circular economy, pollution prevention, biodiversity)
Sectors: 13 (forestry, environment, manufacturing, energy, water supply,
    transport, construction, ICT, professional activities, financial,
    education, health, arts)
Activities: 150+ economic activities across all 6 objectives
KPIs: Turnover, CapEx, OpEx (IAS 1 / IAS 16 / IAS 38 references)
GAR: Stock ratio, Flow ratio, BTAR for credit institutions
"""

__version__ = "1.0.0"
__standard__ = "EU Taxonomy 2020/852 + Climate DA 2021/2139 + Environmental DA 2023/2486"

from .config import TaxonomyAppConfig
from .activity_screening_engine import ActivityScreeningEngine
from .substantial_contribution_engine import SubstantialContributionEngine
from .dnsh_assessment_engine import DNSHAssessmentEngine
from .minimum_safeguards_engine import MinimumSafeguardsEngine
from .kpi_calculation_engine import KPICalculationEngine
from .gar_calculation_engine import GARCalculationEngine
from .alignment_engine import AlignmentEngine
from .reporting_engine import ReportingEngine
from .data_quality_engine import DataQualityEngine
from .regulatory_update_engine import RegulatoryUpdateEngine
from .setup import TaxonomyPlatform, create_app

__all__ = [
    "__version__",
    "__standard__",
    # Configuration
    "TaxonomyAppConfig",
    # Service Engines (10)
    "ActivityScreeningEngine",
    "SubstantialContributionEngine",
    "DNSHAssessmentEngine",
    "MinimumSafeguardsEngine",
    "KPICalculationEngine",
    "GARCalculationEngine",
    "AlignmentEngine",
    "ReportingEngine",
    "DataQualityEngine",
    "RegulatoryUpdateEngine",
    # Platform Facade
    "TaxonomyPlatform",
    "create_app",
]
