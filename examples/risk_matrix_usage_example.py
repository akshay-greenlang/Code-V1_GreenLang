"""
Risk Matrix Implementation Example

Demonstrates usage of the RiskMatrix and RiskRegister classes for
process heat agent safety management per IEC 61511.

Example scenarios:
1. Basic risk assessment and matrix calculation
2. Risk register operations
3. Integration with HAZOP and FMEA results
4. Compliance reporting
5. Risk trending and analytics
"""

from greenlang.safety.risk_matrix import (
    RiskMatrix,
    RiskRegister,
    RiskData,
    RiskLevel,
    RiskCategory,
    RiskStatus,
)
from datetime import datetime, timedelta


def example_1_basic_risk_assessment():
    """Example 1: Calculate risk level from severity and likelihood."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Risk Matrix Calculations")
    print("=" * 70)

    # Test all severity/likelihood combinations
    test_cases = [
        (1, 1, "Minor + Remote = LOW"),
        (2, 3, "Significant + Moderate = MEDIUM"),
        (3, 4, "Serious + Probable = HIGH"),
        (4, 4, "Major + Probable = HIGH"),
        (5, 5, "Catastrophic + Almost Certain = CRITICAL"),
        (4, 5, "Major + Almost Certain = CRITICAL"),
    ]

    for severity, likelihood, description in test_cases:
        risk_level = RiskMatrix.calculate_risk_level(severity, likelihood)
        color = RiskMatrix.get_risk_color(risk_level)
        sil = RiskMatrix.get_required_sil(risk_level)
        days = RiskMatrix.get_acceptance_days(risk_level)

        print(f"\n{description}")
        print(f"  Risk Level: {risk_level.value.upper()}")
        print(f"  Color: {color}")
        print(f"  Required SIL: {sil.value}")
        print(f"  Days to Action: {days}")


def example_2_risk_register_operations():
    """Example 2: Create and manage risks in a register."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Risk Register Operations")
    print("=" * 70)

    register = RiskRegister()

    # Add risks to register
    risks_to_add = [
        RiskData(
            title="Boiler Pressure Relief Valve Failure",
            description="PRV stuck open, pressure cannot be maintained",
            category=RiskCategory.SAFETY,
            severity=5,
            likelihood=2,
            source="HAZOP",
            source_id="DEV-001",
            mitigation_strategy="Install redundant PRV with periodic testing",
            responsible_party="Operations Manager",
        ),
        RiskData(
            title="High Furnace Temperature Excursion",
            description="Uncontrolled temperature rise leads to equipment damage",
            category=RiskCategory.OPERATIONAL,
            severity=4,
            likelihood=3,
            source="HAZOP",
            source_id="DEV-002",
            mitigation_strategy="Install temperature limit control and alarm",
            responsible_party="Control Systems Engineer",
        ),
        RiskData(
            title="Fuel Supply Line Rupture",
            description="Loss of fuel containment, environmental hazard",
            category=RiskCategory.ENVIRONMENTAL,
            severity=4,
            likelihood=2,
            source="FMEA",
            source_id="FM-015",
            mitigation_strategy="Install automatic fuel shutoff valve",
            responsible_party="Maintenance Supervisor",
        ),
        RiskData(
            title="Instrumentation Calibration Drift",
            description="Sensor drift leads to incorrect process readings",
            category=RiskCategory.OPERATIONAL,
            severity=2,
            likelihood=3,
            mitigation_strategy="Implement quarterly calibration schedule",
            responsible_party="Instrument Technician",
        ),
    ]

    print("\nAdding risks to register...")
    for risk_data in risks_to_add:
        added_risk = register.add_risk(risk_data)
        print(f"  Added: {added_risk.risk_id} - {added_risk.title}")
        print(f"         Risk Level: {added_risk.risk_level.value.upper()}")
        print(f"         Required SIL: {added_risk.required_sil.value}")

    # Query risks
    print(f"\nTotal risks in register: {len(register.risks)}")

    critical_risks = register.get_critical_risks()
    print(f"\nCritical risks requiring immediate action: {len(critical_risks)}")
    for risk in critical_risks:
        print(f"  - {risk.title} (Due: {risk.target_mitigation_date.date()})")

    # Update risk status
    print("\nUpdating risk status...")
    first_risk_id = list(register.risks.keys())[0]
    updated = register.update_risk(first_risk_id, {
        "status": RiskStatus.IN_PROGRESS,
        "responsible_party": "Safety Team",
    })
    print(f"  Updated {updated.risk_id} status to IN_PROGRESS")

    # Filter by category
    safety_risks = register.get_open_risks(RiskCategory.SAFETY)
    print(f"\nOpen safety risks: {len(safety_risks)}")


def example_3_hazop_fmea_integration():
    """Example 3: Import risks from HAZOP and FMEA studies."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Integration with HAZOP and FMEA")
    print("=" * 70)

    register = RiskRegister()

    # Simulate HAZOP deviations
    hazop_deviations = [
        {
            "deviation_id": "DEV-001",
            "deviation_description": "NO FLOW through heat exchanger",
            "consequences": ["Loss of heat transfer", "Temperature excursion"],
            "severity": 4,
            "likelihood": 2,
            "recommendations": ["Install flow indicator", "Add low flow alarm"],
        },
        {
            "deviation_id": "DEV-002",
            "deviation_description": "MORE PRESSURE in boiler",
            "consequences": ["Equipment rupture", "Safety hazard"],
            "severity": 5,
            "likelihood": 1,
            "recommendations": ["Install redundant PRV", "Monthly hydrostatic test"],
        },
    ]

    # Simulate FMEA failure modes
    fmea_failure_modes = [
        {
            "fm_id": "FM-001",
            "component_name": "Burner Control System",
            "failure_mode": "Loss of ignition signal",
            "end_effect": "Uncontrolled fuel release",
            "severity": 5,
            "occurrence": 2,
            "rpn": 150,
            "recommended_action": "Implement dual ignition control",
            "responsibility": "Control Systems Team",
        },
        {
            "fm_id": "FM-002",
            "component_name": "Main Fuel Pump",
            "failure_mode": "Bearing wear",
            "end_effect": "Fuel flow reduction",
            "severity": 2,
            "occurrence": 3,
            "rpn": 30,
            "recommended_action": "Implement condition monitoring",
            "responsibility": "Predictive Maintenance Team",
        },
    ]

    print("\nImporting HAZOP deviations...")
    hazop_risks = register.import_from_hazop(hazop_deviations)
    print(f"  Imported {len(hazop_risks)} risks from HAZOP")

    print("\nImporting FMEA failure modes...")
    fmea_risks = register.import_from_fmea(fmea_failure_modes)
    print(f"  Imported {len(fmea_risks)} risks from FMEA")

    print(f"\nTotal risks in register: {len(register.risks)}")


def example_4_heatmap_aggregation():
    """Example 4: Generate risk heatmap and aggregated statistics."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Risk Heatmap and Aggregation")
    print("=" * 70)

    register = RiskRegister()

    # Add sample risks
    for severity in range(1, 6):
        for likelihood in range(1, 6):
            risk = RiskData(
                title=f"Test Risk S{severity}L{likelihood}",
                description="Test risk for heatmap visualization",
                category=RiskCategory.SAFETY,
                severity=severity,
                likelihood=likelihood,
            )
            register.add_risk(risk)

    # Generate heatmap
    risks = list(register.risks.values())
    heatmap = RiskMatrix.generate_heatmap(risks)

    print("\nRisk Heatmap (Count Matrix):")
    print("   Likelihood ->")
    print("   1    2    3    4    5")
    severity_labels = ["1-Minor", "2-Sig", "3-Seri", "4-Mjor", "5-Cata"]
    for severity_idx, row in enumerate(heatmap["matrix"]):
        print(f"{severity_labels[severity_idx]:6}", end="")
        for count in row:
            print(f"{count:4}", end="")
        print()

    # Aggregate statistics
    agg = RiskMatrix.aggregate_risks(risks)

    print("\nRisk Aggregation Statistics:")
    print(f"  Total Open Risks: {agg['total_risks']}")
    print(f"  Critical: {agg['critical']}")
    print(f"  High: {agg['high']}")
    print(f"  Medium: {agg['medium']}")
    print(f"  Low: {agg['low']}")
    print(f"  Average Severity: {agg['average_severity']:.2f}")
    print(f"  Average Likelihood: {agg['average_likelihood']:.2f}")


def example_5_compliance_reporting():
    """Example 5: Generate compliance reports."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Compliance Reporting")
    print("=" * 70)

    register = RiskRegister()

    # Add realistic process heat risks
    boiler_risks = [
        RiskData(
            title="Boiler Feedwater Loss of Control",
            description="Water level uncontrolled, potential for tube damage",
            category=RiskCategory.SAFETY,
            severity=5,
            likelihood=1,
            mitigation_strategy="Triple redundant level measurement and control",
        ),
        RiskData(
            title="Furnace Temperature Overshoot",
            description="Temperature exceeds design limit",
            category=RiskCategory.OPERATIONAL,
            severity=4,
            likelihood=2,
            mitigation_strategy="Install cascade temperature control with multiple sensors",
        ),
        RiskData(
            title="Fuel Gas Odorization Failure",
            description="Unreliable odorant concentration",
            category=RiskCategory.ENVIRONMENTAL,
            severity=3,
            likelihood=2,
            mitigation_strategy="Implement odorant concentration monitoring",
        ),
    ]

    for risk in boiler_risks:
        register.add_risk(risk)

    # Set one risk to overdue for demonstration
    risk_ids = list(register.risks.keys())
    if risk_ids:
        register.update_risk(risk_ids[0], {
            "target_mitigation_date": datetime.utcnow() - timedelta(days=5)
        })

    # Generate report
    report = register.generate_report()

    print("\nRisk Register Summary:")
    print(f"  Generated: {report['report_generated']}")
    print(f"  Total Open Risks: {report['summary']['total_risks']}")
    print(f"  CRITICAL Risks: {report['critical_risks_count']}")
    print(f"  OVERDUE Risks: {report['overdue_risks_count']}")
    print(f"  New Risks (30 days): {report['new_risks_30_days']}")

    print("\nRisks by Category:")
    for category, count in report['category_breakdown'].items():
        print(f"  {category.upper()}: {count}")

    # Export text report
    text_report = register.export_to_compliance_report(format_type="text")
    print("\nCompliance Report (Text):")
    print(text_report[:500] + "...")


def example_6_risk_trending():
    """Example 6: Risk status and trending over time."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Risk Trending and Lifecycle")
    print("=" * 70)

    register = RiskRegister()

    # Create a risk and track its lifecycle
    risk = RiskData(
        title="Combustion Air Damper Calibration",
        description="Damper response drift affects oxygen trim",
        category=RiskCategory.OPERATIONAL,
        severity=3,
        likelihood=2,
        mitigation_strategy="Implement quarterly damper calibration program",
        responsible_party="Instrument Tech",
    )

    print("\nRisk Lifecycle:")
    added_risk = register.add_risk(risk)
    print(f"1. CREATED: {added_risk.risk_id}")
    print(f"   Level: {added_risk.risk_level.value.upper()}")
    print(f"   Target: {added_risk.target_mitigation_date.date()}")

    risk_id = added_risk.risk_id

    # Transition to IN_PROGRESS
    register.update_risk(risk_id, {
        "status": RiskStatus.IN_PROGRESS,
        "mitigation_strategy": "Installing automated damper calibration system",
    })
    print(f"\n2. IN_PROGRESS: Mitigation strategy being implemented")

    # Mark as MITIGATED with residual risk
    register.update_risk(risk_id, {
        "status": RiskStatus.MITIGATED,
        "residual_severity": 1,
        "residual_likelihood": 1,
        "actual_mitigation_date": datetime.utcnow(),
    })

    final_risk = register.risks[risk_id]
    print(f"\n3. MITIGATED: Risk successfully reduced")
    print(f"   Original Risk: S{added_risk.severity}L{added_risk.likelihood}")
    print(f"   Residual Risk: S{final_risk.residual_severity}L{final_risk.residual_likelihood}")
    print(f"   Completion: {final_risk.actual_mitigation_date.date()}")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_risk_assessment()
    example_2_risk_register_operations()
    example_3_hazop_fmea_integration()
    example_4_heatmap_aggregation()
    example_5_compliance_reporting()
    example_6_risk_trending()

    print("\n" + "=" * 70)
    print("RISK MATRIX IMPLEMENTATION EXAMPLE COMPLETE")
    print("=" * 70)
