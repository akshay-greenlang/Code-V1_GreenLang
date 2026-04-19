# -*- coding: utf-8 -*-
"""
GL-016 WATERGUARD - Example Usage and Integration Patterns.

This module demonstrates comprehensive usage examples for the Boiler Water
Treatment Agent, including:
- Basic configuration and setup
- SCADA integration
- Chemical inventory management
- Real-time monitoring
- Multi-boiler coordination
- ERP integration for chemical ordering

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List

from greenlang.GL_016 import (
    BoilerWaterTreatmentAgent,
    AgentConfiguration,
    BoilerConfiguration,
    BoilerType,
    WaterSourceType,
    TreatmentProgramType,
    WaterQualityLimits,
    ChemicalInventory,
    ChemicalSpecification,
    ChemicalType,
    SCADAIntegration,
    ERPIntegration,
    WaterAnalyzerConfiguration,
    ChemicalDosingSystemConfiguration,
    AnalyzerType,
    DosingSystemType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: BASIC SINGLE BOILER CONFIGURATION
# ============================================================================


def example_basic_configuration():
    """
    Example 1: Basic single boiler configuration.

    Demonstrates minimal configuration for a single watertube boiler
    with phosphate treatment program.
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Basic Single Boiler Configuration")
    logger.info("=" * 80)

    # Create boiler configuration
    boiler = BoilerConfiguration(
        boiler_id="BOILER-001",
        boiler_type=BoilerType.WATERTUBE,
        operating_pressure_psig=150,
        operating_temperature_f=366,
        steam_capacity_lb_hr=50000,
        makeup_water_rate_gpm=10,
        condensate_return_pct=85,
        water_source=WaterSourceType.DEMINERALIZED,
        treatment_program=TreatmentProgramType.PHOSPHATE,
        design_cycles_of_concentration=15,
        location="Plant A - Building 1",
    )

    # Create SCADA integration (minimal)
    scada = SCADAIntegration(
        scada_system_name="Plant SCADA",
        protocol="OPC-UA",
        server_address="192.168.1.100",
        server_port=4840,
        authentication_required=True,
        username="waterguard_agent",
    )

    # Create agent configuration
    config = AgentConfiguration(
        agent_id="GL-016",
        agent_name="WATERGUARD",
        boilers=[boiler],
        scada_integration=scada,
        monitoring_interval_seconds=60,
        auto_dosing_enabled=False,  # Manual mode for safety
    )

    logger.info(f"Configuration created for {len(config.boilers)} boiler(s)")
    logger.info(f"Boiler: {boiler.boiler_id} - {boiler.boiler_type.value}")
    logger.info(f"Operating: {boiler.operating_pressure_psig} psig, {boiler.steam_capacity_lb_hr} lb/hr")
    logger.info(f"Treatment: {boiler.treatment_program.value}")

    return config


# ============================================================================
# EXAMPLE 2: COMPREHENSIVE CONFIGURATION WITH SCADA INTEGRATION
# ============================================================================


def example_comprehensive_scada_configuration():
    """
    Example 2: Comprehensive configuration with full SCADA integration.

    Demonstrates complete setup including water analyzers and
    chemical dosing systems.
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Comprehensive SCADA Configuration")
    logger.info("=" * 80)

    # Create boiler configuration
    boiler = BoilerConfiguration(
        boiler_id="BOILER-002",
        boiler_type=BoilerType.WATERTUBE,
        operating_pressure_psig=600,
        operating_temperature_f=490,
        steam_capacity_lb_hr=100000,
        makeup_water_rate_gpm=20,
        condensate_return_pct=90,
        water_source=WaterSourceType.DEMINERALIZED,
        treatment_program=TreatmentProgramType.COORDINATED_PHOSPHATE,
        design_cycles_of_concentration=20,
        boiler_volume_gallons=5000,
        heating_surface_sqft=3000,
        max_tds_ppm=3000,
        max_silica_ppm=90,
        target_ph=10.8,
    )

    # Create water analyzers
    analyzers = [
        WaterAnalyzerConfiguration(
            analyzer_id="ANALYZER-PH-001",
            analyzer_type=AnalyzerType.PH,
            measurement_parameter="pH",
            measurement_units="pH",
            scada_tag="BOILER002.FEEDWATER.PH",
            measurement_range_min=0.0,
            measurement_range_max=14.0,
            accuracy_pct=0.5,
            sampling_interval_seconds=30,
            calibration_interval_days=30,
            location="Feedwater line",
            is_online=True,
        ),
        WaterAnalyzerConfiguration(
            analyzer_id="ANALYZER-COND-001",
            analyzer_type=AnalyzerType.CONDUCTIVITY,
            measurement_parameter="Conductivity",
            measurement_units="µS/cm",
            scada_tag="BOILER002.BOILERWATER.CONDUCTIVITY",
            measurement_range_min=0.0,
            measurement_range_max=10000.0,
            accuracy_pct=1.0,
            sampling_interval_seconds=60,
            calibration_interval_days=90,
            location="Boiler water sampling",
            is_online=True,
        ),
        WaterAnalyzerConfiguration(
            analyzer_id="ANALYZER-DO-001",
            analyzer_type=AnalyzerType.DISSOLVED_OXYGEN,
            measurement_parameter="Dissolved Oxygen",
            measurement_units="ppb",
            scada_tag="BOILER002.FEEDWATER.DO",
            measurement_range_min=0.0,
            measurement_range_max=100.0,
            accuracy_pct=2.0,
            sampling_interval_seconds=60,
            calibration_interval_days=60,
            location="Deaerator outlet",
            is_online=True,
        ),
        WaterAnalyzerConfiguration(
            analyzer_id="ANALYZER-PHOS-001",
            analyzer_type=AnalyzerType.PHOSPHATE,
            measurement_parameter="Phosphate",
            measurement_units="ppm PO4",
            scada_tag="BOILER002.BOILERWATER.PHOSPHATE",
            measurement_range_min=0.0,
            measurement_range_max=100.0,
            accuracy_pct=5.0,
            sampling_interval_seconds=300,
            calibration_interval_days=30,
            location="Boiler water sampling",
            is_online=True,
        ),
    ]

    # Create chemical dosing systems
    dosing_systems = [
        ChemicalDosingSystemConfiguration(
            dosing_system_id="DOSER-PHOSPHATE-001",
            dosing_system_type=DosingSystemType.METERING_PUMP,
            chemical_id="CHEM-001",
            scada_control_tag="BOILER002.CHEMICAL.PHOSPHATE.SETPOINT",
            scada_feedback_tag="BOILER002.CHEMICAL.PHOSPHATE.ACTUAL",
            max_dosing_rate_gph=5.0,
            min_dosing_rate_gph=0.0,
            current_dosing_rate_gph=0.5,
            injection_point="Feedwater line upstream of economizer",
            control_mode="automatic",
            is_online=True,
        ),
        ChemicalDosingSystemConfiguration(
            dosing_system_id="DOSER-SCAVENGER-001",
            dosing_system_type=DosingSystemType.METERING_PUMP,
            chemical_id="CHEM-002",
            scada_control_tag="BOILER002.CHEMICAL.SCAVENGER.SETPOINT",
            scada_feedback_tag="BOILER002.CHEMICAL.SCAVENGER.ACTUAL",
            max_dosing_rate_gph=3.0,
            min_dosing_rate_gph=0.0,
            current_dosing_rate_gph=0.3,
            injection_point="Deaerator inlet",
            control_mode="automatic",
            is_online=True,
        ),
        ChemicalDosingSystemConfiguration(
            dosing_system_id="DOSER-AMINE-001",
            dosing_system_type=DosingSystemType.PROPORTIONAL_FEEDER,
            chemical_id="CHEM-003",
            scada_control_tag="BOILER002.CHEMICAL.AMINE.SETPOINT",
            scada_feedback_tag="BOILER002.CHEMICAL.AMINE.ACTUAL",
            max_dosing_rate_gph=2.0,
            min_dosing_rate_gph=0.0,
            current_dosing_rate_gph=0.2,
            injection_point="Condensate return header",
            control_mode="automatic",
            is_online=True,
        ),
    ]

    # Create SCADA integration
    scada = SCADAIntegration(
        scada_system_name="Plant SCADA System",
        protocol="OPC-UA",
        server_address="192.168.1.100",
        server_port=4840,
        polling_interval_seconds=5,
        timeout_seconds=30,
        authentication_required=True,
        username="waterguard",
        enable_ssl=True,
        water_analyzers=analyzers,
        dosing_systems=dosing_systems,
    )

    # Create agent configuration
    config = AgentConfiguration(
        agent_id="GL-016",
        agent_name="WATERGUARD",
        boilers=[boiler],
        scada_integration=scada,
        monitoring_interval_seconds=60,
        auto_dosing_enabled=True,  # Automatic dosing enabled
        scale_risk_threshold=0.7,
        corrosion_risk_threshold=0.7,
        alert_enabled=True,
    )

    logger.info(f"Comprehensive configuration created:")
    logger.info(f"  - Water Analyzers: {len(scada.water_analyzers)}")
    logger.info(f"  - Dosing Systems: {len(scada.dosing_systems)}")
    logger.info(f"  - Auto-dosing: {'Enabled' if config.auto_dosing_enabled else 'Disabled'}")

    return config


# ============================================================================
# EXAMPLE 3: CHEMICAL INVENTORY MANAGEMENT
# ============================================================================


def example_chemical_inventory_management():
    """
    Example 3: Chemical inventory management.

    Demonstrates chemical inventory tracking and reorder alerts.
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Chemical Inventory Management")
    logger.info("=" * 80)

    # Create chemical specifications
    chemicals = [
        ChemicalSpecification(
            chemical_id="CHEM-001",
            chemical_name="Trisodium Phosphate",
            chemical_type=ChemicalType.PHOSPHATE,
            concentration_pct=30,
            density_lb_gal=10.2,
            current_inventory_gallons=150,
            min_inventory_gallons=50,
            max_inventory_gallons=500,
            unit_cost_usd_gal=12.50,
            supplier="Water Treatment Solutions Inc.",
            msds_number="SDS-TSP-001",
        ),
        ChemicalSpecification(
            chemical_id="CHEM-002",
            chemical_name="Sodium Sulfite",
            chemical_type=ChemicalType.OXYGEN_SCAVENGER,
            concentration_pct=40,
            density_lb_gal=11.5,
            current_inventory_gallons=200,
            min_inventory_gallons=75,
            max_inventory_gallons=400,
            unit_cost_usd_gal=8.75,
            supplier="ChemSupply Corp",
            msds_number="SDS-SULFITE-002",
        ),
        ChemicalSpecification(
            chemical_id="CHEM-003",
            chemical_name="Neutralizing Amine Blend",
            chemical_type=ChemicalType.AMINE,
            concentration_pct=100,
            density_lb_gal=8.3,
            current_inventory_gallons=45,  # Below minimum!
            min_inventory_gallons=50,
            max_inventory_gallons=300,
            unit_cost_usd_gal=15.00,
            supplier="Water Treatment Solutions Inc.",
            msds_number="SDS-AMINE-003",
        ),
        ChemicalSpecification(
            chemical_id="CHEM-004",
            chemical_name="Non-Oxidizing Biocide",
            chemical_type=ChemicalType.BIOCIDE,
            concentration_pct=20,
            density_lb_gal=9.8,
            current_inventory_gallons=80,
            min_inventory_gallons=25,
            max_inventory_gallons=200,
            unit_cost_usd_gal=45.00,
            supplier="BioChem International",
            msds_number="SDS-BIOCIDE-004",
        ),
        ChemicalSpecification(
            chemical_id="CHEM-005",
            chemical_name="Dispersant Polymer",
            chemical_type=ChemicalType.POLYMER,
            concentration_pct=50,
            density_lb_gal=9.5,
            current_inventory_gallons=120,
            min_inventory_gallons=40,
            max_inventory_gallons=250,
            unit_cost_usd_gal=18.50,
            supplier="Polymer Solutions Ltd.",
            msds_number="SDS-POLYMER-005",
        ),
    ]

    # Create chemical inventory
    inventory = ChemicalInventory(chemicals=chemicals)

    # Check inventory status
    logger.info(f"Total chemicals in inventory: {len(inventory.chemicals)}")

    # Check for low inventory
    low_inventory = inventory.get_low_inventory_chemicals()
    if low_inventory:
        logger.warning(f"LOW INVENTORY ALERT: {len(low_inventory)} chemical(s) need reordering")
        for chem in low_inventory:
            logger.warning(
                f"  - {chem.chemical_name}: {chem.current_inventory_gallons} gal "
                f"(min: {chem.min_inventory_gallons} gal)"
            )
            # Calculate days of supply (assume 2 gal/day usage)
            days_supply = chem.days_of_supply(2.0)
            logger.warning(f"    Days of supply: {days_supply:.1f} days")
    else:
        logger.info("All chemicals above minimum inventory levels")

    # Get chemicals by type
    phosphate_chems = inventory.get_chemicals_by_type(ChemicalType.PHOSPHATE)
    logger.info(f"Phosphate chemicals: {len(phosphate_chems)}")
    for chem in phosphate_chems:
        logger.info(f"  - {chem.chemical_name}: {chem.current_inventory_gallons} gal")

    return inventory


# ============================================================================
# EXAMPLE 4: MULTI-BOILER CONFIGURATION
# ============================================================================


def example_multi_boiler_configuration():
    """
    Example 4: Multi-boiler plant configuration.

    Demonstrates configuration for a plant with multiple boilers
    of different types and pressures.
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Multi-Boiler Plant Configuration")
    logger.info("=" * 80)

    # Create multiple boiler configurations
    boilers = [
        BoilerConfiguration(
            boiler_id="BOILER-HP-001",
            boiler_type=BoilerType.WATERTUBE,
            operating_pressure_psig=900,
            operating_temperature_f=530,
            steam_capacity_lb_hr=150000,
            makeup_water_rate_gpm=25,
            condensate_return_pct=95,
            water_source=WaterSourceType.DEMINERALIZED,
            treatment_program=TreatmentProgramType.COORDINATED_PHOSPHATE,
            design_cycles_of_concentration=25,
            location="Plant A - High Pressure",
        ),
        BoilerConfiguration(
            boiler_id="BOILER-LP-001",
            boiler_type=BoilerType.FIRETUBE,
            operating_pressure_psig=150,
            operating_temperature_f=366,
            steam_capacity_lb_hr=40000,
            makeup_water_rate_gpm=8,
            condensate_return_pct=80,
            water_source=WaterSourceType.MUNICIPAL,
            treatment_program=TreatmentProgramType.PHOSPHATE,
            design_cycles_of_concentration=12,
            location="Plant A - Low Pressure",
        ),
        BoilerConfiguration(
            boiler_id="BOILER-WHR-001",
            boiler_type=BoilerType.WASTE_HEAT,
            operating_pressure_psig=250,
            operating_temperature_f=406,
            steam_capacity_lb_hr=60000,
            makeup_water_rate_gpm=12,
            condensate_return_pct=90,
            water_source=WaterSourceType.DEMINERALIZED,
            treatment_program=TreatmentProgramType.ALL_VOLATILE,
            design_cycles_of_concentration=18,
            location="Plant B - Waste Heat Recovery",
        ),
    ]

    # Create SCADA integration
    scada = SCADAIntegration(
        scada_system_name="Multi-Plant SCADA",
        protocol="OPC-UA",
        server_address="192.168.1.100",
        server_port=4840,
        authentication_required=True,
    )

    # Create agent configuration
    config = AgentConfiguration(
        agent_id="GL-016",
        agent_name="WATERGUARD-MULTI",
        boilers=boilers,
        scada_integration=scada,
        monitoring_interval_seconds=30,
        auto_dosing_enabled=False,
    )

    logger.info(f"Multi-boiler configuration created:")
    for boiler in boilers:
        logger.info(
            f"  - {boiler.boiler_id}: {boiler.boiler_type.value}, "
            f"{boiler.operating_pressure_psig} psig, "
            f"{boiler.treatment_program.value}"
        )

    # Show water quality limits for each boiler
    for boiler in boilers:
        limits = config.get_water_quality_limits(boiler.boiler_id)
        logger.info(
            f"  {boiler.boiler_id} Limits: pH {limits.ph_min}-{limits.ph_max}, "
            f"Silica max {limits.silica_max_ppm} ppm"
        )

    return config


# ============================================================================
# EXAMPLE 5: ERP INTEGRATION FOR CHEMICAL ORDERING
# ============================================================================


def example_erp_integration():
    """
    Example 5: ERP integration for automated chemical ordering.

    Demonstrates integration with ERP system for automatic
    chemical reordering and cost tracking.
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 5: ERP Integration")
    logger.info("=" * 80)

    # Create boiler configuration
    boiler = BoilerConfiguration(
        boiler_id="BOILER-003",
        boiler_type=BoilerType.WATERTUBE,
        operating_pressure_psig=450,
        operating_temperature_f=460,
        steam_capacity_lb_hr=75000,
        makeup_water_rate_gpm=15,
        condensate_return_pct=88,
        water_source=WaterSourceType.DEMINERALIZED,
        treatment_program=TreatmentProgramType.PHOSPHATE,
        design_cycles_of_concentration=18,
    )

    # Create SCADA integration
    scada = SCADAIntegration(
        scada_system_name="Plant SCADA",
        protocol="OPC-UA",
        server_address="192.168.1.100",
        server_port=4840,
        authentication_required=True,
    )

    # Create ERP integration
    erp = ERPIntegration(
        erp_system_name="SAP ERP",
        api_endpoint="https://erp.company.com/api/v1",
        api_version="v1",
        authentication_type="oauth2",
        enable_chemical_ordering=True,
        enable_cost_tracking=True,
        enable_maintenance_scheduling=True,
        auto_reorder_threshold_days=14,  # Reorder when 14 days supply remaining
    )

    # Create chemical inventory
    inventory = example_chemical_inventory_management()

    # Create agent configuration
    config = AgentConfiguration(
        agent_id="GL-016",
        agent_name="WATERGUARD",
        boilers=[boiler],
        scada_integration=scada,
        erp_integration=erp,
        chemical_inventory=inventory,
        monitoring_interval_seconds=60,
        auto_dosing_enabled=True,
    )

    logger.info("ERP integration configured:")
    logger.info(f"  - ERP System: {erp.erp_system_name}")
    logger.info(f"  - Auto-ordering: {'Enabled' if erp.enable_chemical_ordering else 'Disabled'}")
    logger.info(f"  - Cost tracking: {'Enabled' if erp.enable_cost_tracking else 'Disabled'}")
    logger.info(f"  - Reorder threshold: {erp.auto_reorder_threshold_days} days")

    return config


# ============================================================================
# EXAMPLE 6: RUNNING THE AGENT
# ============================================================================


async def example_run_agent():
    """
    Example 6: Running the water treatment agent.

    Demonstrates how to instantiate and run the agent with
    real-time monitoring.
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 6: Running Water Treatment Agent")
    logger.info("=" * 80)

    # Create configuration
    config = example_comprehensive_scada_configuration()

    # Create agent instance
    agent = BoilerWaterTreatmentAgent(config)

    logger.info("Agent created successfully")
    logger.info(f"Agent ID: {agent.config.agent_id}")
    logger.info(f"Agent Name: {agent.config.agent_name}")
    logger.info(f"Version: {agent.config.version}")

    try:
        # Execute water treatment analysis
        logger.info("Executing water treatment analysis...")
        result = await agent.execute()

        # Display results
        logger.info("=" * 80)
        logger.info("WATER TREATMENT ANALYSIS RESULTS")
        logger.info("=" * 80)

        logger.info(f"Boiler: {result.boiler_id}")
        logger.info(f"Timestamp: {result.timestamp}")
        logger.info(f"Processing Time: {result.processing_time_seconds:.2f} seconds")
        logger.info(f"Compliance Status: {result.compliance_status}")

        # Water chemistry
        if result.water_chemistry:
            logger.info("\nWATER CHEMISTRY:")
            logger.info(f"  pH: {result.water_chemistry.ph}")
            logger.info(f"  Conductivity: {result.water_chemistry.conductivity_us_cm} µS/cm")
            logger.info(f"  TDS: {result.water_chemistry.total_dissolved_solids_ppm} ppm")
            logger.info(f"  Silica: {result.water_chemistry.silica_ppm} ppm")
            logger.info(f"  Dissolved Oxygen: {result.water_chemistry.dissolved_oxygen_ppb} ppb")
            logger.info(f"  Phosphate: {result.water_chemistry.phosphate_ppm} ppm")

        # Blowdown analysis
        if result.blowdown_analysis:
            logger.info("\nBLOWDOWN ANALYSIS:")
            logger.info(f"  Cycles of Concentration: {result.blowdown_analysis.cycles_of_concentration:.1f}")
            logger.info(f"  Blowdown Percentage: {result.blowdown_analysis.blowdown_percentage:.2f}%")
            logger.info(f"  Surface Blowdown: {result.blowdown_analysis.surface_blowdown_rate_gpm:.2f} GPM")
            logger.info(f"  Bottom Blowdown: {result.blowdown_analysis.bottom_blowdown_rate_gpm:.2f} GPM")

        # Risk assessment
        if result.risk_assessment:
            logger.info("\nRISK ASSESSMENT:")
            logger.info(f"  Scale Risk: {result.risk_assessment.scale_risk_level} ({result.risk_assessment.scale_risk_score:.2f})")
            logger.info(f"  Corrosion Risk: {result.risk_assessment.corrosion_risk_level} ({result.risk_assessment.corrosion_risk_score:.2f})")
            if result.risk_assessment.scale_contributing_factors:
                logger.info(f"  Scale Factors: {', '.join(result.risk_assessment.scale_contributing_factors)}")
            if result.risk_assessment.corrosion_contributing_factors:
                logger.info(f"  Corrosion Factors: {', '.join(result.risk_assessment.corrosion_contributing_factors)}")

        # Chemical optimization
        if result.optimization:
            logger.info("\nCHEMICAL OPTIMIZATION:")
            logger.info(f"  Target pH: {result.optimization.target_ph}")
            logger.info(f"  Target Phosphate: {result.optimization.target_phosphate_ppm} ppm")
            logger.info(f"  Confidence: {result.optimization.confidence_score:.2f}")
            if result.optimization.optimization_rationale:
                logger.info("  Recommendations:")
                for rec in result.optimization.optimization_rationale:
                    logger.info(f"    - {rec}")

        # Compliance violations
        if result.compliance_violations:
            logger.warning("\nCOMPLIANCE VIOLATIONS:")
            for violation in result.compliance_violations:
                logger.warning(f"  - {violation}")

        # Alerts
        if result.alerts:
            logger.warning("\nALERTS:")
            for alert in result.alerts:
                logger.warning(f"  - {alert}")

        # Notifications
        if result.notifications:
            logger.info("\nNOTIFICATIONS:")
            for notification in result.notifications:
                logger.info(f"  - {notification}")

        logger.info(f"\nProvenance Hash: {result.provenance_hash}")

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)

    logger.info("=" * 80)


# ============================================================================
# EXAMPLE 7: CONTINUOUS MONITORING LOOP
# ============================================================================


async def example_continuous_monitoring():
    """
    Example 7: Continuous monitoring loop.

    Demonstrates how to run the agent in continuous monitoring mode.
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 7: Continuous Monitoring Mode")
    logger.info("=" * 80)

    # Create configuration
    config = example_comprehensive_scada_configuration()
    config.monitoring_interval_seconds = 10  # Monitor every 10 seconds

    # Create agent instance
    agent = BoilerWaterTreatmentAgent(config)

    logger.info(f"Starting continuous monitoring (interval: {config.monitoring_interval_seconds}s)")
    logger.info("Press Ctrl+C to stop")

    try:
        cycle_count = 0
        max_cycles = 5  # Run for 5 cycles in this example

        while cycle_count < max_cycles:
            cycle_count += 1
            logger.info(f"\n--- Monitoring Cycle {cycle_count} ---")

            # Execute analysis
            result = await agent.execute()

            # Log key metrics
            logger.info(
                f"Status: {result.compliance_status} | "
                f"pH: {result.water_chemistry.ph if result.water_chemistry else 'N/A'} | "
                f"Cycles: {result.blowdown_analysis.cycles_of_concentration:.1f if result.blowdown_analysis else 'N/A'}"
            )

            # Check for alerts
            if result.alerts:
                logger.warning(f"ALERTS: {len(result.alerts)} active")
                for alert in result.alerts[:3]:  # Show first 3 alerts
                    logger.warning(f"  {alert}")

            # Wait for next monitoring cycle
            if cycle_count < max_cycles:
                await asyncio.sleep(config.monitoring_interval_seconds)

    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring error: {e}", exc_info=True)

    logger.info("Continuous monitoring completed")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    """Main execution function."""
    logger.info("\n" + "=" * 80)
    logger.info("GL-016 WATERGUARD - Example Usage Demonstrations")
    logger.info("=" * 80 + "\n")

    # Run examples
    example_basic_configuration()
    print()

    example_comprehensive_scada_configuration()
    print()

    example_chemical_inventory_management()
    print()

    example_multi_boiler_configuration()
    print()

    example_erp_integration()
    print()

    # Run async examples
    await example_run_agent()
    print()

    await example_continuous_monitoring()

    logger.info("\n" + "=" * 80)
    logger.info("All examples completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Run all examples
    asyncio.run(main())
