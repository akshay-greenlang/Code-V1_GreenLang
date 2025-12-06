"""
NFPA 86 Furnace Compliance Example

This example demonstrates how to use the NFPA86ComplianceChecker to validate
furnace configurations against NFPA 86:2023 requirements for all four furnace
classes (A, B, C, D).

Example scenarios:
1. Class A furnace with flammable volatiles
2. Class B furnace with heated flammable materials
3. Class C furnace with special atmosphere
4. Class D vacuum furnace
"""

from greenlang.safety.nfpa_86_furnace import (
    NFPA86ComplianceChecker,
    FurnaceClass,
    AtmosphereType,
    PurgeConfiguration,
    SafetyInterlockConfig,
    FurnaceConfiguration,
)


def example_class_a_furnace():
    """Example: Check Class A furnace compliance."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Class A Furnace (with flammable volatiles)")
    print("="*70)

    # Create checker
    checker = NFPA86ComplianceChecker()

    # Configure Class A furnace
    furnace_config = FurnaceConfiguration(
        equipment_id="OVEN-CLASS-A-001",
        classification=FurnaceClass.CLASS_A,
        atmosphere_type=AtmosphereType.AIR,
        maximum_temperature_deg_f=1200.0,
        furnace_volume_cubic_feet=800.0,
        burner_input_btuh=800000.0,
        has_flame_supervision=True,
        has_combustion_safeguards=True,
        has_lel_monitoring=True,
        has_emergency_shutdown=True,
        has_temperature_monitoring=True,
        interlocks=[
            SafetyInterlockConfig(
                name="Low Fuel Pressure",
                setpoint=5.0,
                unit="psig",
                is_high_trip=False,
                is_operational=True,
            ),
            SafetyInterlockConfig(
                name="High Temperature",
                setpoint=1200.0,
                unit="°F",
                is_high_trip=True,
                is_operational=True,
            ),
        ],
    )

    # Check compliance
    result = checker.check_class_a_furnace(furnace_config)

    # Display results
    print(f"\nEquipment ID: {result.equipment_id}")
    print(f"Classification: {result.classification.value}")
    print(f"Compliance Level: {result.compliance_level.value}")
    print(f"Compliance Percentage: {result.compliance_percent:.1f}%")
    print(f"Requirements Met: {result.requirements_met}/{result.total_requirements}")

    if result.findings:
        print("\nFindings:")
        for finding in result.findings:
            print(f"  - Section {finding.get('section')}: {finding.get('requirement')}")
    else:
        print("\nNo compliance issues found.")

    # Validate LEL monitoring
    lel_level, lel_msg = checker.validate_lel_monitoring(22.0)
    print(f"\nLEL Monitoring (22%): {lel_msg}")

    # Validate flame failure response
    is_compliant, response_ms, msg = checker.calculate_flame_failure_response(
        detection_time_ms=1.5,
        fuel_shutoff_time_ms=2.0
    )
    print(f"Flame Failure Response: {msg}")


def example_class_c_furnace():
    """Example: Check Class C furnace (special atmosphere) compliance."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Class C Furnace (special atmosphere)")
    print("="*70)

    checker = NFPA86ComplianceChecker()

    # Configure Class C furnace with endothermic atmosphere
    furnace_config = FurnaceConfiguration(
        equipment_id="FURN-CLASS-C-001",
        classification=FurnaceClass.CLASS_C,
        atmosphere_type=AtmosphereType.ENDOTHERMIC,
        maximum_temperature_deg_f=1800.0,
        furnace_volume_cubic_feet=1200.0,
        burner_input_btuh=1500000.0,
        has_purge_capability=True,
        has_lel_monitoring=True,
        has_emergency_shutdown=True,
        has_temperature_monitoring=True,
        purge_config=PurgeConfiguration(
            furnace_volume_cubic_feet=1200.0,
            airflow_cfm=1500.0,
        ),
        interlocks=[
            SafetyInterlockConfig(
                name="Atmosphere Pressure",
                setpoint=0.5,
                unit="inH2O",
                is_operational=True,
            ),
            SafetyInterlockConfig(
                name="Furnace Temperature",
                setpoint=1800.0,
                unit="°F",
                is_high_trip=True,
                is_operational=True,
            ),
        ],
    )

    # Check compliance
    result = checker.check_class_c_furnace(furnace_config)

    print(f"\nEquipment ID: {result.equipment_id}")
    print(f"Classification: {result.classification.value}")
    print(f"Atmosphere Type: {furnace_config.atmosphere_type.value}")
    print(f"Compliance Level: {result.compliance_level.value}")
    print(f"Compliance Percentage: {result.compliance_percent:.1f}%")

    # Validate purge cycle
    is_valid, purge_msg, purge_time = checker.validate_purge_cycle(
        atmosphere=AtmosphereType.ENDOTHERMIC,
        volume_cubic_feet=1200.0,
        flow_rate_cfm=1500.0
    )

    print(f"\nPurge Cycle Validation:")
    print(f"  Valid: {is_valid}")
    print(f"  Message: {purge_msg}")
    print(f"  Required Time: {purge_time:.1f} seconds")

    # Validate trial for ignition
    trial_ok, trial_msg = checker.validate_trial_for_ignition(
        pilot_trial_seconds=8.0,
        main_trial_seconds=6.0
    )
    print(f"\nTrial for Ignition: {trial_msg}")


def example_class_d_furnace():
    """Example: Check Class D furnace (vacuum) compliance."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Class D Furnace (vacuum)")
    print("="*70)

    checker = NFPA86ComplianceChecker()

    # Configure Class D vacuum furnace
    furnace_config = FurnaceConfiguration(
        equipment_id="VAC-FURN-001",
        classification=FurnaceClass.CLASS_D,
        atmosphere_type=AtmosphereType.VACUUM,
        maximum_temperature_deg_f=2400.0,
        furnace_volume_cubic_feet=500.0,
        burner_input_btuh=0.0,  # No direct fuel input in vacuum furnace
        has_emergency_shutdown=True,
        has_temperature_monitoring=True,
        interlocks=[
            SafetyInterlockConfig(
                name="Vacuum Pressure (Low Limit)",
                setpoint=100.0,
                unit="mtorr",
                is_high_trip=False,
                is_operational=True,
            ),
            SafetyInterlockConfig(
                name="Water Cooling Temperature",
                setpoint=130.0,
                unit="°F",
                is_high_trip=True,
                is_operational=True,
            ),
        ],
    )

    # Check compliance
    result = checker.check_class_d_furnace(furnace_config)

    print(f"\nEquipment ID: {result.equipment_id}")
    print(f"Classification: {result.classification.value}")
    print(f"Atmosphere Type: {furnace_config.atmosphere_type.value}")
    print(f"Compliance Level: {result.compliance_level.value}")
    print(f"Compliance Percentage: {result.compliance_percent:.1f}%")

    # Validate safety interlocks
    interlock_valid, interlock_msg = checker.validate_safety_interlocks(
        furnace_config
    )
    print(f"\nSafety Interlocks: {interlock_msg}")


def example_compliance_report():
    """Example: Generate comprehensive compliance report."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Compliance Report Generation")
    print("="*70)

    checker = NFPA86ComplianceChecker()

    # Check multiple furnaces
    configs = [
        FurnaceConfiguration(
            equipment_id="FURN-A-001",
            classification=FurnaceClass.CLASS_A,
            atmosphere_type=AtmosphereType.AIR,
            maximum_temperature_deg_f=1000.0,
            furnace_volume_cubic_feet=500.0,
            burner_input_btuh=500000.0,
            has_flame_supervision=True,
            has_combustion_safeguards=True,
            has_lel_monitoring=True,
            has_emergency_shutdown=True,
            has_temperature_monitoring=True,
            interlocks=[
                SafetyInterlockConfig(name="Interlock-1", is_operational=True)
            ],
        ),
        FurnaceConfiguration(
            equipment_id="FURN-B-001",
            classification=FurnaceClass.CLASS_B,
            atmosphere_type=AtmosphereType.AIR,
            maximum_temperature_deg_f=1200.0,
            furnace_volume_cubic_feet=400.0,
            burner_input_btuh=600000.0,
            has_flame_supervision=True,
            has_combustion_safeguards=True,
            has_emergency_shutdown=True,
            has_temperature_monitoring=True,
            interlocks=[
                SafetyInterlockConfig(name="Interlock-1", is_operational=True)
            ],
        ),
    ]

    # Run checks
    for config in configs:
        if config.classification == FurnaceClass.CLASS_A:
            checker.check_class_a_furnace(config)
        elif config.classification == FurnaceClass.CLASS_B:
            checker.check_class_b_furnace(config)

    # Generate report
    report = checker.get_compliance_report()

    print(f"\nReport ID: {report['report_id']}")
    print(f"Standard: {report['standard']}")
    print(f"Total Checks: {report['total_checks_performed']}")
    print(f"\nChecks by Class:")
    for class_name, count in report['checks_by_class'].items():
        print(f"  {class_name}: {count}")
    print(f"\nCompliance Summary:")
    for level, count in report['compliance_summary'].items():
        print(f"  {level}: {count}")


def example_lel_monitoring():
    """Example: LEL monitoring at different levels."""
    print("\n" + "="*70)
    print("EXAMPLE 5: LEL Monitoring at Different Levels")
    print("="*70)

    checker = NFPA86ComplianceChecker()

    lel_levels = [10.0, 25.0, 35.0, 50.0, 75.0]

    for lel in lel_levels:
        level, message = checker.validate_lel_monitoring(lel)
        print(f"\nLEL {lel:5.1f}%: {level.value:15} - {message}")


if __name__ == "__main__":
    # Run all examples
    example_class_a_furnace()
    example_class_c_furnace()
    example_class_d_furnace()
    example_compliance_report()
    example_lel_monitoring()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")
