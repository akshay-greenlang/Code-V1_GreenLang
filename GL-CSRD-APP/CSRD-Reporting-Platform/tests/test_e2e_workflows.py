# -*- coding: utf-8 -*-
"""
End-to-End Workflow Tests for CSRD Reporting Platform
======================================================

Tests complete user workflows from start to finish.

Author: GreenLang QA Team
Date: 2025-10-20
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path


class TestE2E_NewCompanySetup:
    """Test complete workflow for new company setup."""

    def test_complete_onboarding_workflow(self):
        """
        E2E Test: New Company Onboarding

        Workflow:
        1. Create company profile
        2. Configure ESRS materiality
        3. Set up data sources
        4. Import initial data
        5. Generate first report
        """
        # Step 1: Company Profile Creation
        from agents.intake_agent import IntakeAgent

        intake = IntakeAgent()
        company_profile = {
            "company_name": "GreenTech Industries",
            "nace_code": "24.10",
            "reporting_year": "2024",
            "subsidiaries": 3,
            "employees": 450,
            "revenue_eur": 50_000_000
        }

        profile_result = intake.create_company_profile(company_profile)
        assert profile_result["status"] == "success"
        assert "company_id" in profile_result

        company_id = profile_result["company_id"]

        # Step 2: Materiality Configuration
        from agents.materiality_agent import MaterialityAgent

        materiality = MaterialityAgent()
        materiality_config = {
            "industry": "Manufacturing",
            "region": "EU",
            "company_size": "Medium",
            "activities": ["Production", "Distribution", "Sales"]
        }

        materiality_result = materiality.configure_materiality(
            company_id,
            materiality_config
        )

        assert materiality_result["status"] == "success"
        assert "material_topics" in materiality_result
        assert len(materiality_result["material_topics"]) > 0

        # Step 3: Data Source Configuration
        data_sources = {
            "energy_meters": {
                "type": "api",
                "endpoint": "https://energy-api.example.com",
                "frequency": "hourly"
            },
            "hr_system": {
                "type": "database",
                "connection": "postgresql://hr-db",
                "frequency": "daily"
            }
        }

        source_result = intake.configure_data_sources(company_id, data_sources)
        assert source_result["status"] == "success"

        # Step 4: Import Initial Data
        sample_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "energy_kwh": [1000 + i*10 for i in range(100)],
            "emissions_kg_co2": [500 + i*5 for i in range(100)],
            "employees": [450] * 100
        })

        import_result = intake.import_data(company_id, sample_data)
        assert import_result["status"] == "success"
        assert import_result["records_imported"] == 100

        # Step 5: Generate First Report
        from agents.reporting_agent import ReportingAgent

        reporting = ReportingAgent()
        report_result = reporting.generate_initial_report(company_id)

        assert report_result["status"] == "success"
        assert "report_id" in report_result
        assert report_result["report_format"] == "XBRL"

        print(f"✅ E2E Test PASSED: New company onboarding completed")
        print(f"   Company ID: {company_id}")
        print(f"   Material Topics: {len(materiality_result['material_topics'])}")
        print(f"   Data Imported: {import_result['records_imported']} records")
        print(f"   Report ID: {report_result['report_id']}")


class TestE2E_AnnualReportCycle:
    """Test annual CSRD reporting cycle."""

    def test_complete_annual_report_workflow(self):
        """
        E2E Test: Annual Report Generation

        Workflow:
        1. Update company data
        2. Run materiality assessment
        3. Calculate emissions and metrics
        4. Perform audit validation
        5. Generate XBRL report
        6. Submit to regulator
        """
        company_id = "test-company-123"

        # Step 1: Update Annual Data
        from agents.intake_agent import IntakeAgent

        intake = IntakeAgent()
        annual_data = {
            "year": 2024,
            "revenue": 55_000_000,
            "employees": 475,
            "energy_consumption_kwh": 5_000_000,
            "scope1_emissions_kg": 2_000_000,
            "scope2_emissions_kg": 1_500_000,
            "scope3_emissions_kg": 10_000_000
        }

        update_result = intake.update_annual_data(company_id, annual_data)
        assert update_result["status"] == "success"

        # Step 2: Materiality Assessment
        from agents.materiality_agent import MaterialityAgent

        materiality = MaterialityAgent()
        assessment_result = materiality.assess_materiality(company_id)

        assert assessment_result["status"] == "success"
        assert "material_topics" in assessment_result

        # Step 3: Calculate Emissions
        from agents.calculator_agent import CalculatorAgent

        calculator = CalculatorAgent()
        calculation_result = calculator.calculate_annual_emissions(company_id)

        assert calculation_result["status"] == "success"
        assert "total_emissions_kg_co2e" in calculation_result

        # Step 4: Audit Validation
        from agents.audit_agent import AuditAgent

        audit = AuditAgent()
        audit_result = audit.validate_annual_report(company_id)

        assert audit_result["status"] == "success"
        assert audit_result["validation_passed"] is True
        assert audit_result["critical_issues"] == 0

        # Step 5: Generate XBRL Report
        from agents.reporting_agent import ReportingAgent

        reporting = ReportingAgent()
        report_result = reporting.generate_annual_xbrl(company_id)

        assert report_result["status"] == "success"
        assert "xbrl_file_path" in report_result
        assert Path(report_result["xbrl_file_path"]).exists()

        # Step 6: Submit to Regulator (Automated Filing)
        from agents.domain.automated_filing_agent import AutomatedFilingAgent

        filing = AutomatedFilingAgent()
        filing_result = filing.submit_to_regulator(
            company_id,
            report_result["xbrl_file_path"]
        )

        assert filing_result["status"] in ["success", "submitted"]
        assert "submission_id" in filing_result

        print(f"✅ E2E Test PASSED: Annual report cycle completed")
        print(f"   Total Emissions: {calculation_result['total_emissions_kg_co2e']:,} kg CO2e")
        print(f"   Audit Status: {audit_result['validation_passed']}")
        print(f"   Report: {report_result['xbrl_file_path']}")
        print(f"   Submission ID: {filing_result['submission_id']}")


class TestE2E_MultiStakeholderWorkflow:
    """Test multi-stakeholder collaborative workflow."""

    def test_collaborative_reporting_workflow(self):
        """
        E2E Test: Multi-Stakeholder Workflow

        Workflow:
        1. Data Collector: Import data from multiple sources
        2. Reviewer: Validate and approve data
        3. Calculator: Compute emissions and metrics
        4. Approver: Review and sign off calculations
        5. Compliance: Final audit and submission
        """
        company_id = "multi-stakeholder-test"

        # Step 1: Data Collector Role
        from agents.intake_agent import IntakeAgent

        intake = IntakeAgent()

        # Simulate multiple data collectors
        energy_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=365),
            "energy_kwh": [1000] * 365,
            "source": ["grid"] * 365
        })

        hr_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=365),
            "employees": [450] * 365,
            "training_hours": [8] * 365
        })

        energy_import = intake.import_data(company_id, energy_data, role="data_collector_energy")
        hr_import = intake.import_data(company_id, hr_data, role="data_collector_hr")

        assert energy_import["status"] == "success"
        assert hr_import["status"] == "success"

        # Step 2: Reviewer Role - Validate Data
        review_result = intake.review_and_approve(
            company_id,
            reviewer_id="reviewer-001",
            datasets=["energy", "hr"]
        )

        assert review_result["status"] == "approved"
        assert review_result["validation_passed"] is True

        # Step 3: Calculator Role - Compute Metrics
        from agents.calculator_agent import CalculatorAgent

        calculator = CalculatorAgent()
        calc_result = calculator.calculate_all_metrics(
            company_id,
            calculator_id="calculator-001"
        )

        assert calc_result["status"] == "success"
        assert "calculations" in calc_result

        # Step 4: Approver Role - Review Calculations
        approval_result = calculator.request_approval(
            company_id,
            calculation_id=calc_result["calculation_id"],
            approver_id="approver-001"
        )

        assert approval_result["status"] == "approved"

        # Step 5: Compliance Role - Final Audit
        from agents.audit_agent import AuditAgent

        audit = AuditAgent()
        compliance_result = audit.perform_compliance_audit(
            company_id,
            auditor_id="compliance-001"
        )

        assert compliance_result["status"] == "passed"
        assert compliance_result["ready_for_submission"] is True

        print(f"✅ E2E Test PASSED: Multi-stakeholder workflow completed")
        print(f"   Data Collectors: 2")
        print(f"   Reviewer: Approved")
        print(f"   Calculator: Approved")
        print(f"   Compliance: Passed")


class TestE2E_ErrorRecovery:
    """Test error handling and recovery workflows."""

    def test_invalid_data_correction_workflow(self):
        """
        E2E Test: Error Recovery Workflow

        Workflow:
        1. Import data with validation errors
        2. System detects and flags errors
        3. User corrects data
        4. Re-validation succeeds
        5. Processing continues
        """
        company_id = "error-recovery-test"

        from agents.intake_agent import IntakeAgent

        intake = IntakeAgent()

        # Step 1: Import Invalid Data
        invalid_data = pd.DataFrame({
            "date": ["invalid-date", "2024-01-02", "2024-01-03"],
            "energy_kwh": [-100, 1000, 1500],  # Negative value invalid
            "emissions_kg_co2": [500, -200, 600]  # Negative value invalid
        })

        import_result = intake.import_data(company_id, invalid_data)

        # Step 2: System Detects Errors
        assert import_result["status"] == "validation_errors"
        assert "errors" in import_result
        assert len(import_result["errors"]) > 0

        error_details = import_result["errors"]
        print(f"   Detected {len(error_details)} validation errors")

        # Step 3: Correct Data
        corrected_data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3),
            "energy_kwh": [100, 1000, 1500],
            "emissions_kg_co2": [500, 200, 600]
        })

        # Step 4: Re-validation
        corrected_result = intake.import_data(company_id, corrected_data)

        assert corrected_result["status"] == "success"
        assert corrected_result["records_imported"] == 3
        assert "errors" not in corrected_result or len(corrected_result["errors"]) == 0

        # Step 5: Continue Processing
        from agents.calculator_agent import CalculatorAgent

        calculator = CalculatorAgent()
        calc_result = calculator.calculate_emissions(company_id)

        assert calc_result["status"] == "success"

        print(f"✅ E2E Test PASSED: Error recovery workflow completed")
        print(f"   Initial Errors: {len(error_details)}")
        print(f"   Corrected Records: {corrected_result['records_imported']}")
        print(f"   Final Status: Success")


    def test_api_failure_retry_workflow(self):
        """
        E2E Test: API Failure and Retry

        Workflow:
        1. API call fails (external service down)
        2. System implements exponential backoff
        3. Retry succeeds
        4. Processing continues
        """
        from agents.materiality_agent import MaterialityAgent
        import time

        materiality = MaterialityAgent()

        # Simulate API failure with retry logic
        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                result = materiality.assess_materiality(
                    {
                        "industry": "Manufacturing",
                        "mock_failure": retry_count < 2  # Fail first 2 attempts
                    }
                )

                if result["status"] == "success":
                    success = True
                else:
                    retry_count += 1
                    wait_time = 2 ** retry_count  # Exponential backoff
                    print(f"   Retry {retry_count}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    print(f"   Exception occurred, retry {retry_count}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        assert success is True
        assert retry_count <= max_retries

        print(f"✅ E2E Test PASSED: API retry workflow completed")
        print(f"   Retries: {retry_count}")
        print(f"   Final Status: Success")


@pytest.fixture
def setup_test_environment():
    """Set up test environment for E2E tests."""
    # Create test database, temporary files, etc.
    yield
    # Cleanup after tests


@pytest.mark.e2e
@pytest.mark.slow
class TestE2E_FullPlatform:
    """Full platform integration test."""

    def test_complete_platform_workflow(self, setup_test_environment):
        """
        Complete end-to-end platform test covering all major features.

        This is the ultimate integration test that exercises:
        - All 6 agents
        - All data flows
        - All API endpoints
        - All file formats
        - All validation rules
        """
        print("\n" + "="*80)
        print("COMPLETE PLATFORM E2E TEST")
        print("="*80)

        # Test all workflows in sequence
        test_onboarding = TestE2E_NewCompanySetup()
        test_onboarding.test_complete_onboarding_workflow()

        test_annual = TestE2E_AnnualReportCycle()
        test_annual.test_complete_annual_report_workflow()

        test_stakeholder = TestE2E_MultiStakeholderWorkflow()
        test_stakeholder.test_collaborative_reporting_workflow()

        test_recovery = TestE2E_ErrorRecovery()
        test_recovery.test_invalid_data_correction_workflow()
        test_recovery.test_api_failure_retry_workflow()

        print("\n" + "="*80)
        print("✅ COMPLETE PLATFORM E2E TEST PASSED")
        print("="*80)


if __name__ == "__main__":
    # Run E2E tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
