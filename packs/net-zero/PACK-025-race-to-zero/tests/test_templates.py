# -*- coding: utf-8 -*-
"""
Tests for all 10 PACK-025 Race to Zero Report Templates.

Covers: PledgeCommitmentLetterTemplate, StartingLineChecklistTemplate,
ActionPlanDocumentTemplate, AnnualProgressReportTemplate,
SectorPathwayRoadmapTemplate, PartnershipFrameworkTemplate,
CredibilityAssessmentReportTemplate, CampaignSubmissionPackageTemplate,
DisclosureDashboardTemplate, RaceToZeroCertificateTemplate.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from templates import (
    __version__,
    __pack_id__,
    __pack_name__,
    PledgeCommitmentLetterTemplate,
    StartingLineChecklistTemplate,
    ActionPlanDocumentTemplate,
    AnnualProgressReportTemplate,
    SectorPathwayRoadmapTemplate,
    PartnershipFrameworkTemplate,
    CredibilityAssessmentReportTemplate,
    CampaignSubmissionPackageTemplate,
    DisclosureDashboardTemplate,
    RaceToZeroCertificateTemplate,
)


# ========================================================================
# Module Metadata
# ========================================================================


class TestTemplateModuleMetadata:
    """Tests for templates package metadata."""

    def test_version(self):
        assert __version__ == "1.0.0"

    def test_pack_id(self):
        assert __pack_id__ == "PACK-025"

    def test_pack_name(self):
        assert __pack_name__ == "Race to Zero Pack"


# ========================================================================
# Template 1: PledgeCommitmentLetterTemplate
# ========================================================================


class TestPledgeCommitmentLetterTemplate:
    """Tests for PledgeCommitmentLetterTemplate."""

    def test_template_instantiates(self):
        t = PledgeCommitmentLetterTemplate()
        assert t is not None

    def test_template_has_render(self):
        t = PledgeCommitmentLetterTemplate()
        assert callable(getattr(t, "render_markdown", None))

    def test_template_class_name(self):
        assert PledgeCommitmentLetterTemplate.__name__ == "PledgeCommitmentLetterTemplate"

    def test_template_has_docstring(self):
        assert PledgeCommitmentLetterTemplate.__doc__ is not None


# ========================================================================
# Template 2: StartingLineChecklistTemplate
# ========================================================================


class TestStartingLineChecklistTemplate:
    """Tests for StartingLineChecklistTemplate."""

    def test_template_instantiates(self):
        t = StartingLineChecklistTemplate()
        assert t is not None

    def test_template_has_render(self):
        t = StartingLineChecklistTemplate()
        assert callable(getattr(t, "render_markdown", None))

    def test_template_class_name(self):
        assert StartingLineChecklistTemplate.__name__ == "StartingLineChecklistTemplate"

    def test_template_has_docstring(self):
        assert StartingLineChecklistTemplate.__doc__ is not None


# ========================================================================
# Template 3: ActionPlanDocumentTemplate
# ========================================================================


class TestActionPlanDocumentTemplate:
    """Tests for ActionPlanDocumentTemplate."""

    def test_template_instantiates(self):
        t = ActionPlanDocumentTemplate()
        assert t is not None

    def test_template_has_render(self):
        t = ActionPlanDocumentTemplate()
        assert callable(getattr(t, "render_markdown", None))

    def test_template_class_name(self):
        assert ActionPlanDocumentTemplate.__name__ == "ActionPlanDocumentTemplate"

    def test_template_has_docstring(self):
        assert ActionPlanDocumentTemplate.__doc__ is not None


# ========================================================================
# Template 4: AnnualProgressReportTemplate
# ========================================================================


class TestAnnualProgressReportTemplate:
    """Tests for AnnualProgressReportTemplate."""

    def test_template_instantiates(self):
        t = AnnualProgressReportTemplate()
        assert t is not None

    def test_template_has_render(self):
        t = AnnualProgressReportTemplate()
        assert callable(getattr(t, "render_markdown", None))

    def test_template_class_name(self):
        assert AnnualProgressReportTemplate.__name__ == "AnnualProgressReportTemplate"

    def test_template_has_docstring(self):
        assert AnnualProgressReportTemplate.__doc__ is not None


# ========================================================================
# Template 5: SectorPathwayRoadmapTemplate
# ========================================================================


class TestSectorPathwayRoadmapTemplate:
    """Tests for SectorPathwayRoadmapTemplate."""

    def test_template_instantiates(self):
        t = SectorPathwayRoadmapTemplate()
        assert t is not None

    def test_template_has_render(self):
        t = SectorPathwayRoadmapTemplate()
        assert callable(getattr(t, "render_markdown", None))

    def test_template_class_name(self):
        assert SectorPathwayRoadmapTemplate.__name__ == "SectorPathwayRoadmapTemplate"

    def test_template_has_docstring(self):
        assert SectorPathwayRoadmapTemplate.__doc__ is not None


# ========================================================================
# Template 6: PartnershipFrameworkTemplate
# ========================================================================


class TestPartnershipFrameworkTemplate:
    """Tests for PartnershipFrameworkTemplate."""

    def test_template_instantiates(self):
        t = PartnershipFrameworkTemplate()
        assert t is not None

    def test_template_has_render(self):
        t = PartnershipFrameworkTemplate()
        assert callable(getattr(t, "render_markdown", None))

    def test_template_class_name(self):
        assert PartnershipFrameworkTemplate.__name__ == "PartnershipFrameworkTemplate"

    def test_template_has_docstring(self):
        assert PartnershipFrameworkTemplate.__doc__ is not None


# ========================================================================
# Template 7: CredibilityAssessmentReportTemplate
# ========================================================================


class TestCredibilityAssessmentReportTemplate:
    """Tests for CredibilityAssessmentReportTemplate."""

    def test_template_instantiates(self):
        t = CredibilityAssessmentReportTemplate()
        assert t is not None

    def test_template_has_render(self):
        t = CredibilityAssessmentReportTemplate()
        assert callable(getattr(t, "render_markdown", None))

    def test_template_class_name(self):
        assert CredibilityAssessmentReportTemplate.__name__ == "CredibilityAssessmentReportTemplate"

    def test_template_has_docstring(self):
        assert CredibilityAssessmentReportTemplate.__doc__ is not None


# ========================================================================
# Template 8: CampaignSubmissionPackageTemplate
# ========================================================================


class TestCampaignSubmissionPackageTemplate:
    """Tests for CampaignSubmissionPackageTemplate."""

    def test_template_instantiates(self):
        t = CampaignSubmissionPackageTemplate()
        assert t is not None

    def test_template_has_render(self):
        t = CampaignSubmissionPackageTemplate()
        assert callable(getattr(t, "render_markdown", None))

    def test_template_class_name(self):
        assert CampaignSubmissionPackageTemplate.__name__ == "CampaignSubmissionPackageTemplate"

    def test_template_has_docstring(self):
        assert CampaignSubmissionPackageTemplate.__doc__ is not None


# ========================================================================
# Template 9: DisclosureDashboardTemplate
# ========================================================================


class TestDisclosureDashboardTemplate:
    """Tests for DisclosureDashboardTemplate."""

    def test_template_instantiates(self):
        t = DisclosureDashboardTemplate()
        assert t is not None

    def test_template_has_render(self):
        t = DisclosureDashboardTemplate()
        assert callable(getattr(t, "render_markdown", None))

    def test_template_class_name(self):
        assert DisclosureDashboardTemplate.__name__ == "DisclosureDashboardTemplate"

    def test_template_has_docstring(self):
        assert DisclosureDashboardTemplate.__doc__ is not None


# ========================================================================
# Template 10: RaceToZeroCertificateTemplate
# ========================================================================


class TestRaceToZeroCertificateTemplate:
    """Tests for RaceToZeroCertificateTemplate."""

    def test_template_instantiates(self):
        t = RaceToZeroCertificateTemplate()
        assert t is not None

    def test_template_has_render(self):
        t = RaceToZeroCertificateTemplate()
        assert callable(getattr(t, "render_markdown", None))

    def test_template_class_name(self):
        assert RaceToZeroCertificateTemplate.__name__ == "RaceToZeroCertificateTemplate"

    def test_template_has_docstring(self):
        assert RaceToZeroCertificateTemplate.__doc__ is not None


# ========================================================================
# Cross-Template Tests
# ========================================================================


ALL_TEMPLATE_CLASSES = [
    PledgeCommitmentLetterTemplate,
    StartingLineChecklistTemplate,
    ActionPlanDocumentTemplate,
    AnnualProgressReportTemplate,
    SectorPathwayRoadmapTemplate,
    PartnershipFrameworkTemplate,
    CredibilityAssessmentReportTemplate,
    CampaignSubmissionPackageTemplate,
    DisclosureDashboardTemplate,
    RaceToZeroCertificateTemplate,
]

ALL_TEMPLATE_NAMES = [cls.__name__ for cls in ALL_TEMPLATE_CLASSES]


@pytest.fixture(params=ALL_TEMPLATE_CLASSES, ids=ALL_TEMPLATE_NAMES)
def template_class(request):
    """Parameterized fixture yielding each template class."""
    return request.param


class TestAllTemplatesCommon:
    """Common tests applied to every template class."""

    def test_template_instantiates(self, template_class):
        t = template_class()
        assert t is not None

    def test_template_has_render(self, template_class):
        t = template_class()
        assert callable(getattr(t, "render_markdown", None))

    def test_template_has_docstring(self, template_class):
        assert template_class.__doc__ is not None

    def test_template_name_ends_with_template(self, template_class):
        assert template_class.__name__.endswith("Template")


class TestTemplateCount:
    """Verify all 10 templates are present."""

    def test_all_10_templates_importable(self):
        assert len(ALL_TEMPLATE_CLASSES) == 10

    def test_template_names(self):
        expected = [
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
        assert ALL_TEMPLATE_NAMES == expected
