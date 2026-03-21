# -*- coding: utf-8 -*-
"""
PACK-024 Carbon Neutral Pack - Template Layer
=================================================

10 report templates for carbon neutrality documentation, covering
footprint reports, management plans, credit portfolios, retirement
records, neutralization statements, claims substantiation, verification
packages, annual reports, permanence assessments, and public disclosures.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-024 Carbon Neutral Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-024"
__pack_name__ = "Carbon Neutral Pack"

from .footprint_report import FootprintReportTemplate
from .carbon_mgmt_plan_report import CarbonMgmtPlanReportTemplate
from .credit_portfolio_report import CreditPortfolioReportTemplate
from .registry_retirement_report import RegistryRetirementReportTemplate
from .neutralization_statement_report import NeutralizationStatementReportTemplate
from .claims_substantiation_report import ClaimsSubstantiationReportTemplate
from .verification_package_report import VerificationPackageReportTemplate
from .annual_report import AnnualReportTemplate
from .permanence_assessment_report import PermanenceAssessmentReportTemplate
from .public_disclosure_report import PublicDisclosureReportTemplate

__all__ = [
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "FootprintReportTemplate",
    "CarbonMgmtPlanReportTemplate",
    "CreditPortfolioReportTemplate",
    "RegistryRetirementReportTemplate",
    "NeutralizationStatementReportTemplate",
    "ClaimsSubstantiationReportTemplate",
    "VerificationPackageReportTemplate",
    "AnnualReportTemplate",
    "PermanenceAssessmentReportTemplate",
    "PublicDisclosureReportTemplate",
]
