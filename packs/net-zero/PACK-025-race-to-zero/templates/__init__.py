# -*- coding: utf-8 -*-
"""
PACK-025 Race to Zero Pack - Template Layer
=================================================

10 report templates for Race to Zero campaign documentation, covering
pledge commitment letters, starting line checklists, action plans,
annual progress reports, sector pathway roadmaps, partnership frameworks,
credibility assessments, campaign submission packages, disclosure
dashboards, and verification certificates.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-025 Race to Zero Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-025"
__pack_name__ = "Race to Zero Pack"

from .pledge_commitment_letter import PledgeCommitmentLetterTemplate
from .starting_line_checklist import StartingLineChecklistTemplate
from .action_plan_document import ActionPlanDocumentTemplate
from .annual_progress_report import AnnualProgressReportTemplate
from .sector_pathway_roadmap import SectorPathwayRoadmapTemplate
from .partnership_framework import PartnershipFrameworkTemplate
from .credibility_assessment_report import CredibilityAssessmentReportTemplate
from .campaign_submission_package import CampaignSubmissionPackageTemplate
from .disclosure_dashboard import DisclosureDashboardTemplate
from .race_to_zero_certificate import RaceToZeroCertificateTemplate

__all__ = [
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "PledgeCommitmentLetterTemplate",
    "StartingLineChecklistTemplate",
    "ActionPlanDocumentTemplate",
    "AnnualProgressReportTemplate",
    "SectorPathwayRoadmapTemplate",
    "PartnershipFrameworkTemplate",
    "CredibilityAssessmentReportTemplate",
    "CampaignSubmissionPackageTemplate",
    "DisclosureDashboardTemplate",
    "RaceToZeroCertificateTemplate",
]
