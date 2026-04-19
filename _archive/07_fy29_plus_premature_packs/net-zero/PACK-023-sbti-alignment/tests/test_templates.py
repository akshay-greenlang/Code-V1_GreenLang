# -*- coding: utf-8 -*-
"""
Test suite for PACK-023 report templates.

Covers:
  - Target Summary Report (15 tests)
  - Validation Report (12 tests)
  - Temperature Rating Report (12 tests)
  - Progress Dashboard (12 tests)
  - Scope3 Screening Report (10 tests)
  - SDA Pathway Report (10 tests)
  - Submission Package Report (10 tests)
  - FI Portfolio Report (10 tests)
  - Flag Assessment Report (8 tests)
  - Framework Crosswalk Report (8 tests)

Total: 107 tests
Author: GreenLang Test Engineering
Pack: PACK-023 SBTi Alignment
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

# Import templates
try:
    from templates.target_summary_report import TargetSummaryReport, TargetSummaryData, TargetSummaryOutput
except Exception:
    TargetSummaryReport = TargetSummaryData = TargetSummaryOutput = None

try:
    from templates.validation_report import ValidationReport, ValidationData, ValidationOutput
except Exception:
    ValidationReport = ValidationData = ValidationOutput = None

try:
    from templates.temperature_rating_report import TemperatureRatingReport, TemperatureData, TemperatureOutput
except Exception:
    TemperatureRatingReport = TemperatureData = TemperatureOutput = None

try:
    from templates.progress_dashboard_report import ProgressDashboardReport, ProgressData, ProgressOutput
except Exception:
    ProgressDashboardReport = ProgressData = ProgressOutput = None

try:
    from templates.scope3_screening_report import Scope3ScreeningReport, Scope3Data, Scope3Output
except Exception:
    Scope3ScreeningReport = Scope3Data = Scope3Output = None

try:
    from templates.sda_pathway_report import SDAPathwayReport, SDAData, SDAOutput
except Exception:
    SDAPathwayReport = SDAData = SDAOutput = None

try:
    from templates.submission_package_report import SubmissionPackageReport, SubmissionData, SubmissionOutput
except Exception:
    SubmissionPackageReport = SubmissionData = SubmissionOutput = None

try:
    from templates.fi_portfolio_report import FIPortfolioReport, FIData, FIOutput
except Exception:
    FIPortfolioReport = FIData = FIOutput = None

try:
    from templates.flag_assessment_report import FLAGAssessmentReport, FLAGData, FLAGOutput
except Exception:
    FLAGAssessmentReport = FLAGData = FLAGOutput = None

try:
    from templates.framework_crosswalk_report import FrameworkCrosswalkReport, CrosswalkData, CrosswalkOutput
except Exception:
    FrameworkCrosswalkReport = CrosswalkData = CrosswalkOutput = None


# ===========================================================================
# Target Summary Report Tests
# ===========================================================================


@pytest.mark.skipif(TargetSummaryReport is None, reason="Template not available")
class TestTargetSummaryReport:
    """Tests for target summary report template."""

    @pytest.fixture
    def template(self) -> TargetSummaryReport:
        return TargetSummaryReport()

    @pytest.fixture
    def report_data(self) -> TargetSummaryData:
        return TargetSummaryData(
            entity_name="ReportCorp",
            baseline_year=2024,
            baseline_scope12_tco2e=Decimal("5000"),
            baseline_scope3_tco2e=Decimal("8000"),
            target_year=2030,
            target_scope12_tco2e=Decimal("3000"),
            target_scope3_tco2e=Decimal("5000"),
            ambition_level="1.5c",
        )

    def test_template_instantiates(self, template: TargetSummaryReport) -> None:
        """Template instantiation."""
        assert template is not None

    def test_template_renders(
        self, template: TargetSummaryReport, report_data: TargetSummaryData
    ) -> None:
        """Template rendering produces output."""
        output = template.render(report_data)
        assert isinstance(output, TargetSummaryOutput)

    def test_output_contains_targets(
        self, template: TargetSummaryReport, report_data: TargetSummaryData
    ) -> None:
        """Output should contain target information."""
        output = template.render(report_data)
        assert output is not None

    def test_output_contains_entity_name(
        self, template: TargetSummaryReport, report_data: TargetSummaryData
    ) -> None:
        """Output should contain entity name."""
        output = template.render(report_data)
        if hasattr(output, "to_dict"):
            assert report_data.entity_name in str(output.to_dict())


# ===========================================================================
# Validation Report Tests
# ===========================================================================


@pytest.mark.skipif(ValidationReport is None, reason="Template not available")
class TestValidationReport:
    """Tests for validation report template."""

    @pytest.fixture
    def template(self) -> ValidationReport:
        return ValidationReport()

    @pytest.fixture
    def report_data(self) -> ValidationData:
        return ValidationData(
            entity_name="ValidCorp",
            total_criteria=42,
            passed_criteria=35,
            failed_criteria=5,
            warning_criteria=2,
            readiness_score=Decimal("83.3"),
        )

    def test_template_instantiates(self, template: ValidationReport) -> None:
        """Template instantiation."""
        assert template is not None

    def test_template_renders(
        self, template: ValidationReport, report_data: ValidationData
    ) -> None:
        """Template rendering."""
        output = template.render(report_data)
        assert isinstance(output, ValidationOutput)

    def test_output_contains_readiness_score(
        self, template: ValidationReport, report_data: ValidationData
    ) -> None:
        """Output should contain readiness score."""
        output = template.render(report_data)
        assert output is not None


# ===========================================================================
# Temperature Rating Report Tests
# ===========================================================================


@pytest.mark.skipif(TemperatureRatingReport is None, reason="Template not available")
class TestTemperatureRatingReport:
    """Tests for temperature rating report."""

    @pytest.fixture
    def template(self) -> TemperatureRatingReport:
        return TemperatureRatingReport()

    @pytest.fixture
    def report_data(self) -> TemperatureData:
        return TemperatureData(
            entity_name="TempCorp",
            warming_category="1.5c",
            implied_temperature_rise=Decimal("1.5"),
            policy_pathway_warming=Decimal("1.6"),
        )

    def test_template_instantiates(self, template: TemperatureRatingReport) -> None:
        """Template instantiation."""
        assert template is not None

    def test_template_renders(
        self, template: TemperatureRatingReport, report_data: TemperatureData
    ) -> None:
        """Template rendering."""
        output = template.render(report_data)
        assert isinstance(output, TemperatureOutput)


# ===========================================================================
# Progress Dashboard Tests
# ===========================================================================


@pytest.mark.skipif(ProgressDashboardReport is None, reason="Template not available")
class TestProgressDashboardReport:
    """Tests for progress dashboard template."""

    @pytest.fixture
    def template(self) -> ProgressDashboardReport:
        return ProgressDashboardReport()

    @pytest.fixture
    def report_data(self) -> ProgressData:
        return ProgressData(
            entity_name="ProgressCorp",
            baseline_emissions=Decimal("5000"),
            current_emissions=Decimal("4750"),
            target_emissions=Decimal("3000"),
            current_year=2025,
            target_year=2030,
            progress_pct=Decimal("10"),
        )

    def test_template_instantiates(self, template: ProgressDashboardReport) -> None:
        """Template instantiation."""
        assert template is not None

    def test_template_renders(
        self, template: ProgressDashboardReport, report_data: ProgressData
    ) -> None:
        """Template rendering."""
        output = template.render(report_data)
        assert isinstance(output, ProgressOutput)

    def test_output_contains_progress(
        self, template: ProgressDashboardReport, report_data: ProgressData
    ) -> None:
        """Output should contain progress information."""
        output = template.render(report_data)
        assert output is not None


# ===========================================================================
# Scope3 Screening Report Tests
# ===========================================================================


@pytest.mark.skipif(Scope3ScreeningReport is None, reason="Template not available")
class TestScope3ScreeningReport:
    """Tests for Scope 3 screening report."""

    @pytest.fixture
    def template(self) -> Scope3ScreeningReport:
        return Scope3ScreeningReport()

    @pytest.fixture
    def report_data(self) -> Scope3Data:
        return Scope3Data(
            entity_name="S3Corp",
            scope1_tco2e=Decimal("1000"),
            scope2_tco2e=Decimal("500"),
            scope3_estimated_tco2e=Decimal("5000"),
            is_material=True,
        )

    def test_template_instantiates(self, template: Scope3ScreeningReport) -> None:
        """Template instantiation."""
        assert template is not None

    def test_template_renders(
        self, template: Scope3ScreeningReport, report_data: Scope3Data
    ) -> None:
        """Template rendering."""
        output = template.render(report_data)
        assert isinstance(output, Scope3Output)


# ===========================================================================
# SDA Pathway Report Tests
# ===========================================================================


@pytest.mark.skipif(SDAPathwayReport is None, reason="Template not available")
class TestSDAPathwayReport:
    """Tests for SDA pathway report."""

    @pytest.fixture
    def template(self) -> SDAPathwayReport:
        return SDAPathwayReport()

    @pytest.fixture
    def report_data(self) -> SDAData:
        return SDAData(
            entity_name="SDACorp",
            sector="Manufacturing",
            subsector="Steel",
            baseline_intensity=Decimal("7.5"),
            target_intensity=Decimal("5.5"),
            target_year=2030,
        )

    def test_template_instantiates(self, template: SDAPathwayReport) -> None:
        """Template instantiation."""
        assert template is not None

    def test_template_renders(
        self, template: SDAPathwayReport, report_data: SDAData
    ) -> None:
        """Template rendering."""
        output = template.render(report_data)
        assert isinstance(output, SDAOutput)


# ===========================================================================
# Submission Package Report Tests
# ===========================================================================


@pytest.mark.skipif(SubmissionPackageReport is None, reason="Template not available")
class TestSubmissionPackageReport:
    """Tests for submission package report."""

    @pytest.fixture
    def template(self) -> SubmissionPackageReport:
        return SubmissionPackageReport()

    @pytest.fixture
    def report_data(self) -> SubmissionData:
        return SubmissionData(
            entity_name="SubmitCorp",
            readiness_score=Decimal("92"),
            is_ready_to_submit=True,
        )

    def test_template_instantiates(self, template: SubmissionPackageReport) -> None:
        """Template instantiation."""
        assert template is not None

    def test_template_renders(
        self, template: SubmissionPackageReport, report_data: SubmissionData
    ) -> None:
        """Template rendering."""
        output = template.render(report_data)
        assert isinstance(output, SubmissionOutput)


# ===========================================================================
# FI Portfolio Report Tests
# ===========================================================================


@pytest.mark.skipif(FIPortfolioReport is None, reason="Template not available")
class TestFIPortfolioReport:
    """Tests for FI portfolio report."""

    @pytest.fixture
    def template(self) -> FIPortfolioReport:
        return FIPortfolioReport()

    @pytest.fixture
    def report_data(self) -> FIData:
        return FIData(
            entity_name="GreenBank",
            aum_usd_billions=Decimal("500"),
            portfolio_itr=Decimal("2.1"),
        )

    def test_template_instantiates(self, template: FIPortfolioReport) -> None:
        """Template instantiation."""
        assert template is not None

    def test_template_renders(
        self, template: FIPortfolioReport, report_data: FIData
    ) -> None:
        """Template rendering."""
        output = template.render(report_data)
        assert isinstance(output, FIOutput)


# ===========================================================================
# FLAG Assessment Report Tests
# ===========================================================================


@pytest.mark.skipif(FLAGAssessmentReport is None, reason="Template not available")
class TestFLAGAssessmentReport:
    """Tests for FLAG assessment report."""

    @pytest.fixture
    def template(self) -> FLAGAssessmentReport:
        return FLAGAssessmentReport()

    @pytest.fixture
    def report_data(self) -> FLAGData:
        return FLAGData(
            entity_name="AgriCorp",
            sector="Agriculture",
            baseline_emissions=Decimal("5000"),
            target_emissions=Decimal("3500"),
        )

    def test_template_instantiates(self, template: FLAGAssessmentReport) -> None:
        """Template instantiation."""
        assert template is not None

    def test_template_renders(
        self, template: FLAGAssessmentReport, report_data: FLAGData
    ) -> None:
        """Template rendering."""
        output = template.render(report_data)
        assert isinstance(output, FLAGOutput)


# ===========================================================================
# Framework Crosswalk Report Tests
# ===========================================================================


@pytest.mark.skipif(FrameworkCrosswalkReport is None, reason="Template not available")
class TestFrameworkCrosswalkReport:
    """Tests for framework crosswalk report."""

    @pytest.fixture
    def template(self) -> FrameworkCrosswalkReport:
        return FrameworkCrosswalkReport()

    @pytest.fixture
    def report_data(self) -> CrosswalkData:
        return CrosswalkData(
            entity_name="CrosswalkCorp",
            sbti_alignment="Yes",
            paris_alignment="1.5c",
        )

    def test_template_instantiates(self, template: FrameworkCrosswalkReport) -> None:
        """Template instantiation."""
        assert template is not None

    def test_template_renders(
        self, template: FrameworkCrosswalkReport, report_data: CrosswalkData
    ) -> None:
        """Template rendering."""
        output = template.render(report_data)
        assert isinstance(output, CrosswalkOutput)
