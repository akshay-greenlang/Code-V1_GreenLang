"""
EPA Part 60 NSPS Compliance Checker - Example Usage

This example demonstrates how to use the NSPSComplianceChecker to verify
compliance with EPA 40 CFR Part 60 emission standards for process heat
equipment (boilers, heaters, furnaces).

Standards Covered:
    - Subpart D: Fossil-fuel-fired steam generators (>100 MMBtu/hr)
    - Subpart Db: Industrial boilers (10-100 MMBtu/hr)
    - Subpart Dc: Small boilers and process heaters (<10 MMBtu/hr)
    - Subpart J: Petroleum refinery furnaces

Features Demonstrated:
    1. Multi-subpart compliance checking
    2. F-factor calculations for emission normalization
    3. Compliance margin analysis
    4. Comprehensive reporting
    5. Provenance tracking with SHA-256 hashing
"""

from greenlang.compliance.epa import (
    NSPSComplianceChecker,
    FuelType,
    BoilerType,
    EmissionsData,
    FacilityData,
    FFactorCalculator,
)


def example_1_large_natural_gas_boiler():
    """
    Example 1: Large natural gas steam generator (Subpart D).

    A 150 MMBtu/hr fossil-fuel-fired steam generator burning natural gas
    with measured emissions. Verify compliance with Subpart D limits.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Large Natural Gas Steam Generator (Subpart D)")
    print("=" * 80)

    # Initialize checker
    checker = NSPSComplianceChecker()

    # Define facility
    facility = FacilityData(
        facility_id="PLANT-001",
        equipment_id="BOILER-001",
        boiler_type=BoilerType.FOSSIL_FUEL_STEAM,
        fuel_type=FuelType.NATURAL_GAS,
        heat_input_mmbtu_hr=150.0,
        installation_date="2015-06-15",
        last_stack_test_date="2024-03-20",
        permit_limits={
            "SO2": 0.020,
            "NOx": 0.50,
            "PM": 0.03,
            "Opacity": 20.0,
        },
        continuous_monitoring=True,
    )

    # Measured emissions data
    emissions = EmissionsData(
        so2_ppm=12.5,
        so2_lb_mmbtu=0.018,  # Below 0.020 limit
        nox_ppm=32.0,
        nox_lb_mmbtu=0.46,   # Below 0.50 limit
        pm_gr_dscf=0.025,
        opacity_pct=18.0,    # Below 20% limit
        o2_pct=3.2,
        co2_pct=7.5,
    )

    # Check compliance with Subpart D
    result = checker.check_subpart_d(facility, emissions)

    # Display results
    print(f"\nFacility: {facility.facility_id}")
    print(f"Equipment: {facility.equipment_id}")
    print(f"Fuel Type: {facility.fuel_type.value}")
    print(f"Heat Input: {facility.heat_input_mmbtu_hr} MMBtu/hr")
    print(f"\nCompliance Status: {result.compliance_status.value.upper()}")
    print(f"Standard: {result.applicable_subpart}")

    print("\n--- SO2 Compliance ---")
    print(f"  Limit: {result.so2_limit_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  Measured: {result.so2_measured_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  Status: {result.so2_status}")
    print(f"  Margin: {result.so2_compliance_margin:.1f}%")

    print("\n--- NOx Compliance ---")
    print(f"  Limit: {result.nox_limit_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  Measured: {result.nox_measured_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  Status: {result.nox_status}")
    print(f"  Margin: {result.nox_compliance_margin:.1f}%")

    print("\n--- PM Compliance ---")
    print(f"  Measured: {result.pm_measured_gr_dscf:.3f} gr/dscf")
    print(f"  Status: {result.pm_status}")

    print("\n--- Opacity Compliance ---")
    print(f"  Limit: {result.opacity_limit_pct:.1f}%")
    print(f"  Measured: {result.opacity_measured_pct:.1f}%")
    print(f"  Status: {result.opacity_status}")

    if result.findings:
        print("\n--- Findings ---")
        for finding in result.findings:
            print(f"  • {finding}")

    if result.recommendations:
        print("\n--- Recommendations ---")
        for rec in result.recommendations:
            print(f"  • {rec}")

    print(f"\n--- Provenance ---")
    print(f"  Hash: {result.provenance_hash}")
    print(f"  Processing Time: {result.processing_time_ms:.2f} ms")


def example_2_industrial_boiler_noncompliant():
    """
    Example 2: Industrial boiler with exceedance (Subpart Db).

    A 75 MMBtu/hr oil-fired industrial boiler showing NOx exceedance.
    Verify failure and get recommendations.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Industrial Boiler with NOx Exceedance (Subpart Db)")
    print("=" * 80)

    checker = NSPSComplianceChecker()

    facility = FacilityData(
        facility_id="PLANT-002",
        equipment_id="BOILER-002",
        boiler_type=BoilerType.INDUSTRIAL_BOILER,
        fuel_type=FuelType.DISTILLATE_OIL,
        heat_input_mmbtu_hr=75.0,
        continuous_monitoring=False,
    )

    emissions = EmissionsData(
        so2_lb_mmbtu=0.28,   # Compliant (limit 0.30)
        nox_lb_mmbtu=0.35,   # EXCEEDANCE (limit 0.30 for oil)
        pm_lb_mmbtu=0.013,   # Compliant (limit 0.015)
        opacity_pct=22.0,    # EXCEEDANCE (limit 20%)
        o2_pct=3.5,
    )

    result = checker.check_subpart_db(facility, emissions)

    print(f"\nFacility: {facility.facility_id}")
    print(f"Equipment: {facility.equipment_id}")
    print(f"Fuel Type: {facility.fuel_type.value}")
    print(f"Heat Input: {facility.heat_input_mmbtu_hr} MMBtu/hr")
    print(f"\nCompliance Status: {result.compliance_status.value.upper()}")

    print("\n--- NOx Compliance ---")
    print(f"  Limit: {result.nox_limit_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  Measured: {result.nox_measured_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  Status: {result.nox_status}")
    print(f"  Margin: {result.nox_compliance_margin:.1f}% (NEGATIVE = EXCEEDANCE)")

    print("\n--- Opacity Compliance ---")
    print(f"  Limit: {result.opacity_limit_pct:.1f}%")
    print(f"  Measured: {result.opacity_measured_pct:.1f}%")
    print(f"  Status: {result.opacity_status}")

    print("\n--- Findings ---")
    for finding in result.findings:
        print(f"  ALERT: {finding}")

    print("\n--- Required Actions ---")
    for rec in result.recommendations:
        print(f"  ACTION: {rec}")


def example_3_coal_fired_subpart_d():
    """
    Example 3: Coal-fired steam generator (Subpart D).

    A 200 MMBtu/hr coal-fired boiler with higher emission limits compared
    to natural gas due to fuel type.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Coal-Fired Steam Generator (Subpart D)")
    print("=" * 80)

    checker = NSPSComplianceChecker()

    facility = FacilityData(
        facility_id="PLANT-003",
        equipment_id="BOILER-003",
        boiler_type=BoilerType.FOSSIL_FUEL_STEAM,
        fuel_type=FuelType.COAL,  # Coal has higher SO2 limit (0.30 vs 0.020)
        heat_input_mmbtu_hr=200.0,
    )

    emissions = EmissionsData(
        so2_lb_mmbtu=0.25,   # Below 0.30 limit for coal
        nox_lb_mmbtu=0.48,   # Below 0.50 limit
        pm_gr_dscf=0.026,
        opacity_pct=19.0,
        o2_pct=3.8,
    )

    result = checker.check_subpart_d(facility, emissions)

    print(f"\nFacility: {facility.facility_id} (COAL-FIRED)")
    print(f"Heat Input: {facility.heat_input_mmbtu_hr} MMBtu/hr")
    print(f"\nCompliance Status: {result.compliance_status.value.upper()}")

    print("\n--- SO2 Compliance (Note: Coal Has Higher Limit) ---")
    print(f"  Coal SO2 Limit: {result.so2_limit_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  (Compare to natural gas: 0.020 lb/MMBtu)")
    print(f"  Measured: {result.so2_measured_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  Status: {result.so2_status}")
    print(f"  Margin: {result.so2_compliance_margin:.1f}%")

    print("\n--- Key Point ---")
    print("  Coal-fired units have relaxed SO2 limits compared to gas/oil")
    print("  due to the fuel's inherent sulfur content characteristics.")


def example_4_petroleum_refinery():
    """
    Example 4: Petroleum refinery fuel gas furnace (Subpart J).

    A furnace at a petroleum refinery burning fuel gas with unique
    stricter opacity limits and CO monitoring.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Petroleum Refinery Furnace (Subpart J)")
    print("=" * 80)

    checker = NSPSComplianceChecker()

    facility = FacilityData(
        facility_id="REFINERY-001",
        equipment_id="FURNACE-001",
        boiler_type=BoilerType.PROCESS_HEATER,
        fuel_type=FuelType.COAL_DERIVED,  # Fuel gas
        heat_input_mmbtu_hr=85.0,
        continuous_monitoring=True,
    )

    emissions = EmissionsData(
        nox_lb_mmbtu=0.28,   # Below 0.30 limit
        co_lb_mmbtu=0.65,    # Above 0.60 guideline
        pm_lb_mmbtu=0.012,
        opacity_pct=4.5,     # Stricter limit: 5% vs 20% for boilers
        o2_pct=2.8,
    )

    result = checker.check_subpart_j(facility, emissions)

    print(f"\nFacility: {facility.facility_id}")
    print(f"Equipment: {facility.equipment_id} (Refinery Furnace)")
    print(f"\nCompliance Status: {result.compliance_status.value.upper()}")

    print("\n--- Subpart J Specific Standards ---")
    print(f"  NOx Limit: {result.nox_limit_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  NOx Measured: {result.nox_measured_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  Status: {result.nox_status}")

    print("\n--- CO Monitoring (Guideline) ---")
    print(f"  CO Measured: {emissions.co_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  (Guideline: 0.60 lb/MMBtu)")

    print("\n--- Stricter Opacity Limit ---")
    print(f"  Subpart J Limit: {result.opacity_limit_pct:.1f}% (15-min avg)")
    print(f"  vs Subpart D: 20% (6-min avg)")
    print(f"  Measured: {result.opacity_measured_pct:.1f}%")
    print(f"  Status: {result.opacity_status}")

    print("\n--- Findings ---")
    for finding in result.findings:
        print(f"  • {finding}")


def example_5_small_boiler_subpart_dc():
    """
    Example 5: Small boiler with relaxed limits (Subpart Dc).

    A 7 MMBtu/hr natural gas boiler subject to Subpart Dc limits
    which are more relaxed than Subpart Db for industrial boilers.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Small Boiler with Relaxed Limits (Subpart Dc)")
    print("=" * 80)

    checker = NSPSComplianceChecker()

    facility = FacilityData(
        facility_id="SMALL-001",
        equipment_id="BOILER-SMALL-001",
        boiler_type=BoilerType.SMALL_BOILER,
        fuel_type=FuelType.NATURAL_GAS,
        heat_input_mmbtu_hr=7.0,
    )

    emissions = EmissionsData(
        so2_lb_mmbtu=0.025,
        nox_lb_mmbtu=0.070,
        pm_lb_mmbtu=0.018,
        opacity_pct=18.0,
        o2_pct=3.5,
    )

    result = checker.check_subpart_dc(facility, emissions)

    print(f"\nFacility: {facility.facility_id}")
    print(f"Heat Input: {facility.heat_input_mmbtu_hr} MMBtu/hr (Small Boiler)")
    print(f"\nCompliance Status: {result.compliance_status.value.upper()}")

    print("\n--- Subpart Dc Relaxed Limits ---")
    print(f"  SO2 Limit: {result.so2_limit_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  (vs Subpart Db: 0.020 lb/MMBtu for gas)")
    print(f"  NOx Limit: {result.nox_limit_lb_mmbtu:.3f} lb/MMBtu")
    print(f"  (vs Subpart Db: 0.060 lb/MMBtu for gas)")
    print(f"  PM Limit: {result.pm_limit_gr_dscf:.3f} gr/dscf")
    print(f"  (vs Subpart Db: 0.020 gr/dscf)")


def example_6_f_factor_calculations():
    """
    Example 6: F-factor calculations for emission normalization (EPA Method 19).

    Demonstrate F-factor calculations (Fd, Fc, Fw) used for normalizing
    emission rates to standard conditions.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: F-Factor Calculations (EPA Method 19)")
    print("=" * 80)

    print("\n--- Fd: SO2 F-Factor (Sulfur Correction) ---")
    print("Fd = (F_base - 250*S) / (1 - S)")

    # Natural gas (no sulfur)
    fd_gas = FFactorCalculator.calculate_fd("natural_gas", so2_fraction=0.0)
    print(f"\nNatural Gas (0% sulfur): Fd = {fd_gas:.2f}")

    # Coal with varying sulfur
    for sulfur_pct in [1.0, 2.0, 3.0, 5.0]:
        fd_coal = FFactorCalculator.calculate_fd("coal_bituminous", so2_fraction=sulfur_pct/100)
        print(f"Coal ({sulfur_pct}% sulfur): Fd = {fd_coal:.2f}")

    print("\n--- Fc: Oxygen Correction Factor ---")
    print("Fc = (20.9 - M) / (20.9 - M_ref), where M = measured O2")

    for o2_pct in [1.5, 3.0, 5.0, 7.0]:
        fc = FFactorCalculator.calculate_fc(excess_o2_pct=o2_pct)
        print(f"  {o2_pct}% O2: Fc = {fc:.3f}")

    print("\n--- Fw: Moisture Correction Factor ---")
    print("Fw = (1 + M) / (1 + M_ref)")

    for moisture_pct in [0.0, 2.5, 5.0, 10.0]:
        fw = FFactorCalculator.calculate_fw("coal_bituminous", moisture_pct=moisture_pct)
        print(f"  {moisture_pct}% moisture: Fw = {fw:.3f}")


def example_7_compliance_report_generation():
    """
    Example 7: Generate comprehensive compliance report.

    Create a detailed compliance report for a facility with all findings,
    recommendations, and provenance tracking.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Comprehensive Compliance Report Generation")
    print("=" * 80)

    checker = NSPSComplianceChecker()

    facility = FacilityData(
        facility_id="PLANT-004",
        equipment_id="BOILER-004",
        boiler_type=BoilerType.INDUSTRIAL_BOILER,
        fuel_type=FuelType.NATURAL_GAS,
        heat_input_mmbtu_hr=50.0,
        installation_date="2018-11-01",
        last_stack_test_date="2024-01-15",
        continuous_monitoring=True,
    )

    emissions = EmissionsData(
        so2_lb_mmbtu=0.015,
        nox_lb_mmbtu=0.055,
        pm_lb_mmbtu=0.012,
        opacity_pct=16.5,
        o2_pct=3.2,
    )

    # Generate comprehensive report
    report = checker.generate_compliance_report(
        facility,
        emissions,
        include_recommendations=True,
    )

    print("\n=== COMPLIANCE REPORT ===\n")
    print(f"Facility ID: {report['facility_id']}")
    print(f"Equipment ID: {report['equipment_id']}")
    print(f"Check Date: {report['check_date']}")
    print(f"Compliance Status: {report['compliance_status'].upper()}")
    print(f"Fuel Type: {report['fuel_type']}")
    print(f"Heat Input: {report['heat_input_mmbtu_hr']} MMBtu/hr")
    print(f"Standard: {report['applicable_standard']}")

    print("\n=== EMISSION STANDARDS COMPLIANCE ===\n")

    print("SO2:")
    print(f"  Status: {report['so2_compliance']['status']}")
    print(f"  Limit: {report['so2_compliance']['limit_lb_mmbtu']:.3f} lb/MMBtu")
    print(f"  Measured: {report['so2_compliance']['measured_lb_mmbtu']:.3f} lb/MMBtu")
    print(f"  Margin: {report['so2_compliance']['margin_pct']:.1f}%")

    print("\nNOx:")
    print(f"  Status: {report['nox_compliance']['status']}")
    print(f"  Limit: {report['nox_compliance']['limit_lb_mmbtu']:.3f} lb/MMBtu")
    print(f"  Measured: {report['nox_compliance']['measured_lb_mmbtu']:.3f} lb/MMBtu")
    print(f"  Margin: {report['nox_compliance']['margin_pct']:.1f}%")

    print("\nPM:")
    print(f"  Status: {report['pm_compliance']['status']}")
    print(f"  Limit: {report['pm_compliance']['limit_gr_dscf']:.2f} gr/dscf")
    print(f"  Measured: {report['pm_compliance']['measured_gr_dscf']:.3f} gr/dscf")
    print(f"  Margin: {report['pm_compliance']['margin_pct']:.1f}%")

    print("\n=== AUDIT TRAIL ===\n")
    print(f"Provenance Hash (SHA-256): {report['provenance_hash']}")
    print(f"Processing Time: {report['processing_time_ms']:.2f} ms")


if __name__ == "__main__":
    print("\n")
    print("#" * 80)
    print("# EPA PART 60 NSPS COMPLIANCE CHECKER - EXAMPLES")
    print("#" * 80)

    example_1_large_natural_gas_boiler()
    example_2_industrial_boiler_noncompliant()
    example_3_coal_fired_subpart_d()
    example_4_petroleum_refinery()
    example_5_small_boiler_subpart_dc()
    example_6_f_factor_calculations()
    example_7_compliance_report_generation()

    print("\n" + "#" * 80)
    print("# END OF EXAMPLES")
    print("#" * 80 + "\n")
