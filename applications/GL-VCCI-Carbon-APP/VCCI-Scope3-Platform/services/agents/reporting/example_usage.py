# -*- coding: utf-8 -*-
"""
Scope3ReportingAgent Example Usage
GL-VCCI Scope 3 Platform

Comprehensive examples demonstrating all agent capabilities.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""

from datetime import datetime
from services.agents.reporting import (
    Scope3ReportingAgent,
    CompanyInfo,
    EmissionsData,
    EnergyData,
    IntensityMetrics,
    RisksOpportunities,
    TransportData,
    ValidationLevel,
)


def example_1_basic_esrs_e1():
    """Example 1: Basic ESRS E1 report generation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic ESRS E1 Report")
    print("=" * 80)

    # Initialize agent
    agent = Scope3ReportingAgent()

    # Company information
    company_info = CompanyInfo(
        name="Acme Manufacturing Corp",
        reporting_year=2024,
        headquarters="Munich, Germany",
        number_of_employees=5000,
        annual_revenue_usd=500_000_000,
        industry_sector="Manufacturing",
    )

    # Emissions data
    emissions_data = EmissionsData(
        scope1_tco2e=1234.5,
        scope2_location_tco2e=2345.6,
        scope2_market_tco2e=1890.3,
        scope3_tco2e=20000.0,
        scope3_categories={
            1: 15000.0,  # Purchased Goods & Services
            4: 3000.0,   # Upstream Transportation
            6: 2000.0,   # Business Travel
        },
        avg_dqi_score=85.5,
        data_quality_by_scope={
            "Scope 1": 92.0,
            "Scope 2": 95.0,
            "Scope 3": 80.0,
        },
        reporting_period_start=datetime(2024, 1, 1),
        reporting_period_end=datetime(2024, 12, 31),
    )

    # Generate report
    result = agent.generate_esrs_e1_report(
        emissions_data=emissions_data,
        company_info=company_info,
        export_format="json",
        output_path="output/esrs_e1_report_2024.json",
    )

    print(f"\n✅ ESRS E1 Report Generated Successfully!")
    print(f"   📁 File: {result.file_path}")
    print(f"   📊 Charts: {result.charts_count}")
    print(f"   📋 Tables: {result.tables_count}")
    print(f"   ✓ Validation: {'PASSED' if result.validation_result.is_valid else 'FAILED'}")
    print(f"   📈 Data Quality: {result.metadata.data_quality_score:.1f}/100")


def example_2_cdp_questionnaire():
    """Example 2: CDP questionnaire auto-population."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: CDP Questionnaire Auto-Population")
    print("=" * 80)

    agent = Scope3ReportingAgent()

    company_info = CompanyInfo(
        name="Tech Innovations Inc",
        reporting_year=2024,
    )

    emissions_data = EmissionsData(
        scope1_tco2e=500.0,
        scope2_location_tco2e=1200.0,
        scope2_market_tco2e=1000.0,
        scope3_tco2e=12000.0,
        scope3_categories={
            1: 8000.0,
            4: 2500.0,
            6: 1500.0,
        },
        avg_dqi_score=88.0,
        reporting_period_start=datetime(2024, 1, 1),
        reporting_period_end=datetime(2024, 12, 31),
    )

    energy_data = EnergyData(
        total_energy_mwh=5000.0,
        renewable_energy_mwh=2000.0,
        non_renewable_energy_mwh=3000.0,
        renewable_pct=40.0,
    )

    result = agent.generate_cdp_report(
        emissions_data=emissions_data,
        company_info=company_info,
        energy_data=energy_data,
        export_format="json",
        output_path="output/cdp_questionnaire_2024.json",
    )

    print(f"\n✅ CDP Questionnaire Generated!")
    print(f"   📁 File: {result.file_path}")
    print(f"   📈 Auto-Population Rate: {result.content.get('auto_population_rate', 0):.0%}")
    print(f"   📋 Sections: {', '.join(result.sections_generated)}")


def example_3_ifrs_s2_with_risks():
    """Example 3: IFRS S2 report with climate risks and opportunities."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: IFRS S2 Climate Disclosures")
    print("=" * 80)

    agent = Scope3ReportingAgent()

    company_info = CompanyInfo(
        name="Global Energy Solutions",
        reporting_year=2024,
    )

    emissions_data = EmissionsData(
        scope1_tco2e=3000.0,
        scope2_location_tco2e=5000.0,
        scope2_market_tco2e=4500.0,
        scope3_tco2e=30000.0,
        scope3_categories={
            1: 20000.0,
            4: 6000.0,
            6: 4000.0,
        },
        avg_dqi_score=82.0,
        reporting_period_start=datetime(2024, 1, 1),
        reporting_period_end=datetime(2024, 12, 31),
    )

    # Climate risks and opportunities
    risks_opportunities = RisksOpportunities(
        physical_risks=[
            {
                "type": "Acute",
                "description": "Increased frequency of extreme weather events",
                "impact": "Medium",
                "timeframe": "Short-term",
            }
        ],
        transition_risks=[
            {
                "type": "Policy",
                "description": "Carbon pricing mechanisms",
                "impact": "High",
                "timeframe": "Medium-term",
            }
        ],
        opportunities=[
            {
                "type": "Products and Services",
                "description": "Development of low-carbon products",
                "impact": "High",
                "timeframe": "Medium-term",
            }
        ],
        financial_impact_assessment="Climate risks could impact 15-20% of revenue by 2030.",
    )

    result = agent.generate_ifrs_s2_report(
        emissions_data=emissions_data,
        company_info=company_info,
        risks_opportunities=risks_opportunities,
        export_format="json",
    )

    print(f"\n✅ IFRS S2 Report Generated!")
    print(f"   📁 File: {result.file_path}")
    print(f"   🌍 Physical Risks: {len(risks_opportunities.physical_risks)}")
    print(f"   🔄 Transition Risks: {len(risks_opportunities.transition_risks)}")
    print(f"   💡 Opportunities: {len(risks_opportunities.opportunities)}")


def example_4_iso_14083_transport():
    """Example 4: ISO 14083 transport conformance certificate."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: ISO 14083 Transport Conformance")
    print("=" * 80)

    agent = Scope3ReportingAgent()

    transport_data = TransportData(
        transport_by_mode={
            "road": {
                "emissions_tco2e": 1500.0,
                "tonne_km": 50000,
                "emission_factor": 0.030,
            },
            "sea": {
                "emissions_tco2e": 1200.0,
                "tonne_km": 80000,
                "emission_factor": 0.015,
            },
            "air": {
                "emissions_tco2e": 300.0,
                "tonne_km": 5000,
                "emission_factor": 0.060,
            },
            "rail": {
                "emissions_tco2e": 200.0,
                "tonne_km": 25000,
                "emission_factor": 0.008,
            },
        },
        total_tonne_km=160000,
        total_emissions_tco2e=3200.0,
        emission_factors_used=[
            {"mode": "road", "factor": 0.030, "source": "DEFRA 2024", "vintage": 2024},
            {"mode": "sea", "factor": 0.015, "source": "GLEC Framework 2024", "vintage": 2024},
            {"mode": "air", "factor": 0.060, "source": "ICAO 2024", "vintage": 2024},
            {"mode": "rail", "factor": 0.008, "source": "UIC 2024", "vintage": 2024},
        ],
        data_quality_score=92.0,
        methodology="ISO 14083:2023",
    )

    result = agent.generate_iso_14083_certificate(
        transport_data=transport_data,
        output_path="output/iso_14083_certificate.json",
    )

    print(f"\n✅ ISO 14083 Certificate Generated!")
    print(f"   📁 File: {result.file_path}")
    print(f"   🔖 Certificate ID: {result.content['certificate_id']}")
    print(f"   🚛 Transport Modes: {', '.join(result.content['transport_modes'])}")
    print(f"   📊 Data Quality: {transport_data.data_quality_score:.1f}/100")


def example_5_comprehensive_with_validation():
    """Example 5: Comprehensive workflow with pre-validation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Comprehensive Workflow with Validation")
    print("=" * 80)

    agent = Scope3ReportingAgent()

    company_info = CompanyInfo(
        name="Sustainable Industries Ltd",
        reporting_year=2024,
        headquarters="London, UK",
        number_of_employees=10000,
        annual_revenue_usd=1_000_000_000,
        industry_sector="Retail",
    )

    emissions_data = EmissionsData(
        scope1_tco2e=2500.0,
        scope2_location_tco2e=4000.0,
        scope2_market_tco2e=3500.0,
        scope3_tco2e=45000.0,
        scope3_categories={
            1: 30000.0,
            2: 5000.0,
            3: 2000.0,
            4: 4000.0,
            6: 3000.0,
            7: 1000.0,
        },
        avg_dqi_score=90.0,
        data_quality_by_scope={
            "Scope 1": 95.0,
            "Scope 2": 98.0,
            "Scope 3": 85.0,
        },
        reporting_period_start=datetime(2024, 1, 1),
        reporting_period_end=datetime(2024, 12, 31),
        prior_year_emissions={
            "scope1_tco2e": 2800.0,
            "scope2_tco2e": 4200.0,
            "scope3_tco2e": 47000.0,
            "total_tco2e": 54000.0,
        },
        yoy_change_pct=-4.6,  # 4.6% reduction
    )

    energy_data = EnergyData(
        total_energy_mwh=15000.0,
        renewable_energy_mwh=7500.0,
        non_renewable_energy_mwh=7500.0,
        renewable_pct=50.0,
    )

    intensity_metrics = IntensityMetrics(
        tco2e_per_million_usd=51.5,
        tco2e_per_fte=5.15,
    )

    # Step 1: Validate data readiness
    print("\n📋 Step 1: Validating data readiness...")
    validation = agent.validate_readiness(
        emissions_data=emissions_data,
        standard="esrs_e1",
        company_info=company_info,
        energy_data=energy_data,
    )

    print(f"\n   Validation Results:")
    print(f"   ✓ Status: {'PASSED' if validation.is_valid else 'FAILED'}")
    print(f"   ✓ Passed Checks: {validation.passed_checks}/{len(validation.checks)}")
    print(f"   ⚠ Warnings: {validation.warnings}")
    print(f"   📊 Completeness: {validation.completeness_pct:.1f}%")

    if validation.is_valid:
        # Step 2: Generate report
        print("\n📄 Step 2: Generating ESRS E1 report...")
        result = agent.generate_esrs_e1_report(
            emissions_data=emissions_data,
            company_info=company_info,
            energy_data=energy_data,
            intensity_metrics=intensity_metrics,
            export_format="json",
        )

        print(f"\n✅ Report Generated Successfully!")
        print(f"   📁 File: {result.file_path}")
        print(f"   📊 Charts: {result.charts_count}")
        print(f"   📋 Tables: {result.tables_count}")
        print(f"   📈 Data Quality: {result.metadata.data_quality_score:.1f}/100")
        print(f"   📉 YoY Change: {emissions_data.yoy_change_pct:+.1f}%")
    else:
        print("\n❌ Validation failed. Address issues before generating report.")
        for check in validation.checks:
            if check.status == "FAIL":
                print(f"   - {check.check_name}: {check.message}")


def example_6_strict_validation():
    """Example 6: Strict validation mode."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Strict Validation Mode")
    print("=" * 80)

    # Initialize agent with strict validation
    agent_strict = Scope3ReportingAgent(config={
        "validation_level": ValidationLevel.STRICT
    })

    company_info = CompanyInfo(
        name="Test Corp",
        reporting_year=2024,
    )

    # Intentionally incomplete data
    emissions_data = EmissionsData(
        scope1_tco2e=1000.0,
        scope2_location_tco2e=2000.0,
        scope2_market_tco2e=1800.0,
        scope3_tco2e=10000.0,
        scope3_categories={
            1: 10000.0,  # Only 1 category (not ideal)
        },
        avg_dqi_score=65.0,  # Below threshold
        reporting_period_start=datetime(2024, 1, 1),
        reporting_period_end=datetime(2024, 12, 31),
    )

    try:
        result = agent_strict.generate_esrs_e1_report(
            emissions_data=emissions_data,
            company_info=company_info,
            export_format="json",
        )
        print("\n✅ Report generated (validation passed)")
    except Exception as e:
        print(f"\n❌ Report generation failed: {str(e)}")
        print("\n   This demonstrates strict validation rejecting low-quality data.")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Scope3ReportingAgent - Comprehensive Examples")
    print("GL-VCCI Scope 3 Platform v1.0.0")
    print("=" * 80)

    try:
        example_1_basic_esrs_e1()
        example_2_cdp_questionnaire()
        example_3_ifrs_s2_with_risks()
        example_4_iso_14083_transport()
        example_5_comprehensive_with_validation()
        example_6_strict_validation()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
