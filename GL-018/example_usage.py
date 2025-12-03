# -*- coding: utf-8 -*-
"""
GL-018 FLUEFLOW - Example Usage Script.

This script demonstrates how to configure and use the FlueGasAnalyzerAgent
for comprehensive flue gas analysis, combustion efficiency optimization,
and emissions compliance monitoring.

Author: GreenLang Team
Date: December 2025
"""

import asyncio
import logging
from datetime import datetime

from greenlang.GL_018 import (
    FlueGasAnalyzerAgent,
    AgentConfiguration,
)
from greenlang.GL_018.config import (
    BurnerConfiguration,
    BurnerType,
    FuelType,
    FuelSpecification,
    EmissionsStandard,
    EmissionsLimits,
    AnalyzerType,
    ControlStrategy,
    SCADAIntegration,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def create_example_configuration() -> AgentConfiguration:
    """
    Create example agent configuration for a natural gas boiler burner.

    Returns:
        AgentConfiguration with complete settings
    """
    logger.info("Creating example configuration for natural gas burner")

    # Define fuel specification (natural gas)
    fuel_spec = FuelSpecification(
        fuel_type=FuelType.NATURAL_GAS,
        fuel_name="Pipeline Natural Gas",
        higher_heating_value_btu_scf=1020,
        lower_heating_value_btu_scf=920,
        carbon_pct=74.0,
        hydrogen_pct=24.0,
        sulfur_pct=0.01,
        nitrogen_pct=1.5,
        oxygen_pct=0.0,
        moisture_pct=0.5,
        specific_gravity=0.60,
        stoichiometric_air_fuel_ratio=17.2,
        theoretical_co2_max_pct=11.7,
    )

    # Define emissions limits (EPA NSPS for natural gas)
    emissions_limits = EmissionsLimits(
        emissions_standard=EmissionsStandard.EPA_NSPS,
        nox_limit_ppm=30.0,
        nox_limit_lb_mmbtu=0.036,
        co_limit_ppm=400.0,
        so2_limit_ppm=50.0,
        pm_limit_mg_m3=20.0,
        reference_o2_pct=3.0,
        opacity_limit_pct=20.0,
        averaging_period_hours=1.0,
        max_exceedances_per_24h=0,
        max_exceedance_duration_minutes=0.0,
    )

    # Define burner configuration
    burner_config = BurnerConfiguration(
        burner_id="BURNER-001",
        burner_type=BurnerType.LOW_NOX,
        burner_manufacturer="Cleaver-Brooks",
        burner_model="ClearFire-H",
        design_firing_rate_mmbtu_hr=60.0,
        minimum_firing_rate_mmbtu_hr=15.0,
        turndown_ratio=4.0,
        fuel_specification=fuel_spec,
        emissions_standard=EmissionsStandard.EPA_NSPS,
        emissions_limits=emissions_limits,
        control_strategy=ControlStrategy.O2_TRIM,
        has_vfd_fan=True,
        has_modulating_fuel_valve=True,
        analyzer_type=AnalyzerType.CEMS,
        has_o2_analyzer=True,
        has_co_analyzer=True,
        has_nox_analyzer=True,
        has_co2_analyzer=True,
        has_opacity_monitor=True,
        stack_height_ft=80.0,
        stack_diameter_inches=24.0,
        max_stack_temperature_f=650,
        min_o2_pct=2.0,
        max_o2_pct=8.0,
        target_o2_pct=3.0,
        design_combustion_efficiency_pct=83.0,
        minimum_acceptable_efficiency_pct=78.0,
        location="Building A - Boiler Room",
        commissioning_date=datetime(2020, 6, 15),
        last_tuning_date=datetime(2024, 10, 1),
    )

    # Define SCADA integration
    scada_integration = SCADAIntegration(
        enabled=True,
        scada_system="Wonderware System Platform",
        connection_string="opc.tcp://scada-server:4840",
        polling_interval_seconds=60,
        o2_tag="BOILER.BURNER001.FG_O2_PCT",
        co2_tag="BOILER.BURNER001.FG_CO2_PCT",
        co_tag="BOILER.BURNER001.FG_CO_PPM",
        nox_tag="BOILER.BURNER001.FG_NOX_PPM",
        so2_tag="BOILER.BURNER001.FG_SO2_PPM",
        stack_temp_tag="BOILER.BURNER001.STACK_TEMP",
        fuel_flow_tag="BOILER.BURNER001.FUEL_FLOW",
        air_flow_tag="BOILER.BURNER001.AIR_FLOW",
        fd_fan_speed_tag="BOILER.BURNER001.FD_FAN_SPEED",
        id_fan_speed_tag="BOILER.BURNER001.ID_FAN_SPEED",
        firing_rate_tag="BOILER.BURNER001.FIRING_RATE",
        steam_flow_tag="BOILER.BURNER001.STEAM_FLOW",
        fd_fan_setpoint_tag="BOILER.BURNER001.FD_FAN_SP",
        fuel_valve_setpoint_tag="BOILER.BURNER001.FUEL_VALVE_SP",
        damper_position_setpoint_tag="BOILER.BURNER001.DAMPER_POS_SP",
        enable_data_validation=True,
        max_data_age_seconds=300,
    )

    # Create agent configuration
    agent_config = AgentConfiguration(
        agent_name="GL-018 FLUEFLOW - Building A",
        version="1.0.0",
        environment="production",
        burners=[burner_config],
        scada_integration=scada_integration,
        analysis_interval_seconds=60,
        auto_optimization_enabled=False,  # Set to True for automatic optimization
        optimization_deadband_pct=2.0,
        enable_email_alerts=True,
        enable_sms_alerts=False,
        alert_recipients=[
            "plant.engineer@company.com",
            "operations.manager@company.com",
        ],
        enable_hourly_reports=True,
        enable_daily_reports=True,
        enable_monthly_reports=True,
        report_recipients=[
            "environmental.compliance@company.com",
            "plant.manager@company.com",
        ],
        historical_data_retention_days=90,
        enable_predictive_maintenance=True,
        enable_efficiency_trending=True,
        enable_emissions_forecasting=False,
    )

    return agent_config


async def run_flue_gas_analysis():
    """
    Execute flue gas analysis workflow.

    This function demonstrates the complete workflow:
    1. Create configuration
    2. Initialize agent
    3. Execute analysis
    4. Display results
    """
    logger.info("=" * 80)
    logger.info("GL-018 FLUEFLOW - Flue Gas Analyzer Agent")
    logger.info("=" * 80)

    # Step 1: Create configuration
    logger.info("\n[Step 1] Creating agent configuration...")
    config = create_example_configuration()
    logger.info(f"Configuration created for {len(config.burners)} burner(s)")

    # Step 2: Initialize agent
    logger.info("\n[Step 2] Initializing FlueGasAnalyzerAgent...")
    agent = FlueGasAnalyzerAgent(config)
    logger.info(f"Agent initialized: {config.agent_name} v{config.version}")

    # Step 3: Execute analysis
    logger.info("\n[Step 3] Executing flue gas analysis...")
    result = await agent.execute()
    logger.info(f"Analysis completed in {result.processing_time_seconds:.2f} seconds")

    # Step 4: Display results
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS RESULTS")
    logger.info("=" * 80)

    # System status
    logger.info(f"\n[System Status]")
    logger.info(f"  Overall Status: {result.system_status}")
    logger.info(f"  Performance Status: {result.performance_status}")

    # Flue gas composition
    if result.flue_gas_composition:
        fg = result.flue_gas_composition
        logger.info(f"\n[Flue Gas Composition]")
        logger.info(f"  O2: {fg.oxygen_pct:.2f}%")
        logger.info(f"  CO2: {fg.carbon_dioxide_pct:.2f}%")
        logger.info(f"  CO: {fg.carbon_monoxide_ppm:.0f} ppm")
        logger.info(f"  NOx: {fg.nitrogen_oxides_ppm:.0f} ppm")
        logger.info(f"  SO2: {fg.sulfur_dioxide_ppm:.0f} ppm")
        logger.info(f"  Stack Temperature: {fg.stack_temperature_f:.0f}°F")
        logger.info(f"  Excess Air: {fg.excess_air_pct:.1f}%")

    # Combustion analysis
    if result.combustion_analysis:
        ca = result.combustion_analysis
        logger.info(f"\n[Combustion Analysis]")
        logger.info(f"  Combustion Efficiency: {ca.combustion_efficiency_pct:.2f}%")
        logger.info(f"  Thermal Efficiency: {ca.thermal_efficiency_pct:.2f}%")
        logger.info(f"  Stack Loss: {ca.stack_loss_pct:.2f}%")
        logger.info(f"  Dry Gas Loss: {ca.dry_gas_loss_pct:.2f}%")
        logger.info(f"  Moisture Loss: {ca.moisture_loss_pct:.2f}%")
        logger.info(f"  Unburned Combustibles Loss: {ca.unburned_combustibles_loss_pct:.2f}%")
        logger.info(f"  Excess Air: {ca.excess_air_pct:.1f}%")
        logger.info(f"  Tuning Status: {ca.tuning_status}")
        logger.info(f"  Quality Index: {ca.combustion_quality_index:.3f}")

        if ca.optimization_opportunities:
            logger.info(f"  Optimization Opportunities:")
            for opp in ca.optimization_opportunities:
                logger.info(f"    - {opp}")

    # Efficiency assessment
    if result.efficiency_assessment:
        ea = result.efficiency_assessment
        logger.info(f"\n[Efficiency Assessment]")
        logger.info(f"  Current Efficiency: {ea.current_efficiency_pct:.2f}%")
        logger.info(f"  Baseline Efficiency: {ea.baseline_efficiency_pct:.2f}%")
        logger.info(f"  Deviation: {ea.efficiency_deviation_pct:+.2f}%")
        logger.info(f"  Performance Rating: {ea.performance_rating}")
        logger.info(f"  Performance Score: {ea.performance_score:.0f}/100")
        logger.info(f"  Fuel Cost: ${ea.fuel_cost_per_hour_usd:,.2f}/hour")
        logger.info(f"  Annual Fuel Cost: ${ea.annual_fuel_cost_usd:,.0f}/year")
        if ea.potential_savings_usd > 0:
            logger.info(f"  Potential Savings: ${ea.potential_savings_usd:,.0f}/year")

        if ea.identified_issues:
            logger.info(f"  Identified Issues:")
            for issue in ea.identified_issues:
                logger.info(f"    - {issue}")

    # Air-fuel ratio recommendation
    if result.air_fuel_recommendation:
        afr = result.air_fuel_recommendation
        logger.info(f"\n[Air-Fuel Ratio Optimization]")
        logger.info(f"  Current A/F Ratio: {afr.current_air_fuel_ratio:.2f}")
        logger.info(f"  Target A/F Ratio: {afr.target_air_fuel_ratio:.2f}")
        logger.info(f"  Current Excess Air: {afr.current_excess_air_pct:.1f}%")
        logger.info(f"  Target Excess Air: {afr.target_excess_air_pct:.1f}%")
        logger.info(f"  Current O2: {afr.current_oxygen_pct:.2f}%")
        logger.info(f"  Target O2: {afr.target_oxygen_pct:.2f}%")
        logger.info(f"  Priority: {afr.adjustment_priority}")
        logger.info(f"  Expected Efficiency Gain: {afr.expected_efficiency_gain_pct:.2f}%")

        if abs(afr.air_flow_adjustment_pct) > 0.1:
            logger.info(f"  Recommended Air Flow Adjustment: {afr.air_flow_adjustment_pct:+.1f}%")
        if afr.recommended_fd_fan_speed_pct > 0:
            logger.info(f"  Recommended FD Fan Speed: {afr.recommended_fd_fan_speed_pct:.1f}%")

        logger.info(f"  Rationale: {afr.adjustment_rationale}")

    # Emissions compliance
    if result.emissions_compliance:
        ec = result.emissions_compliance
        logger.info(f"\n[Emissions Compliance]")
        logger.info(f"  Overall Status: {ec.overall_compliance_status}")
        logger.info(f"  Emissions Standard: {ec.emissions_standard}")
        logger.info(f"  Reference O2: {ec.nox_limit_ppm and 3.0}%")

        logger.info(f"\n  NOx:")
        logger.info(f"    Measured: {ec.nox_ppm_corrected:.1f} ppm (corrected)")
        logger.info(f"    Limit: {ec.nox_limit_ppm:.1f} ppm")
        logger.info(f"    Status: {ec.nox_compliance_status}")
        logger.info(f"    Margin to Limit: {ec.nox_margin_to_limit_pct:.1f}%")

        logger.info(f"  CO:")
        logger.info(f"    Measured: {ec.co_ppm_corrected:.1f} ppm (corrected)")
        logger.info(f"    Limit: {ec.co_limit_ppm:.1f} ppm")
        logger.info(f"    Status: {ec.co_compliance_status}")
        logger.info(f"    Margin to Limit: {ec.co_margin_to_limit_pct:.1f}%")

        logger.info(f"  SO2:")
        logger.info(f"    Measured: {ec.so2_ppm_corrected:.1f} ppm (corrected)")
        logger.info(f"    Limit: {ec.so2_limit_ppm:.1f} ppm")
        logger.info(f"    Status: {ec.so2_compliance_status}")

        if ec.violations:
            logger.warning(f"\n  VIOLATIONS:")
            for violation in ec.violations:
                logger.warning(f"    - {violation}")

        if ec.corrective_actions:
            logger.info(f"  Required Corrective Actions:")
            for action in ec.corrective_actions:
                logger.info(f"    - {action}")

    # Optimization recommendations
    if result.optimization_recommendations:
        logger.info(f"\n[Optimization Recommendations]")
        for i, rec in enumerate(result.optimization_recommendations, 1):
            logger.info(f"  {i}. {rec}")

        if result.estimated_savings_usd_per_year > 0:
            logger.info(f"\n  Estimated Annual Savings: ${result.estimated_savings_usd_per_year:,.0f}")

    # Alerts and warnings
    if result.alerts:
        logger.warning(f"\n[ALERTS]")
        for alert in result.alerts:
            logger.warning(f"  - {alert}")

    if result.warnings:
        logger.info(f"\n[Warnings]")
        for warning in result.warnings:
            logger.info(f"  - {warning}")

    # Data provenance
    logger.info(f"\n[Data Provenance]")
    logger.info(f"  Provenance Hash: {result.provenance_hash}")
    logger.info(f"  Data Sources: {', '.join(result.data_sources)}")
    logger.info(f"  Processing Time: {result.processing_time_seconds:.3f} seconds")

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)

    return result


async def continuous_monitoring_example():
    """
    Example of continuous monitoring mode.

    This demonstrates running the agent in a continuous loop
    with periodic analysis execution.
    """
    logger.info("\n" + "=" * 80)
    logger.info("CONTINUOUS MONITORING MODE")
    logger.info("=" * 80)

    config = create_example_configuration()
    agent = FlueGasAnalyzerAgent(config)

    logger.info(f"Starting continuous monitoring with {config.analysis_interval_seconds}s interval")
    logger.info("Press Ctrl+C to stop...\n")

    try:
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration} ---")

            result = await agent.execute()

            # Log key metrics
            if result.combustion_analysis and result.emissions_compliance:
                logger.info(
                    f"Status: {result.system_status} | "
                    f"Efficiency: {result.combustion_analysis.combustion_efficiency_pct:.1f}% | "
                    f"Emissions: {result.emissions_compliance.overall_compliance_status} | "
                    f"O2: {result.flue_gas_composition.oxygen_pct:.1f}% | "
                    f"CO: {result.flue_gas_composition.carbon_monoxide_ppm:.0f} ppm"
                )

            # Wait for next interval
            await asyncio.sleep(config.analysis_interval_seconds)

    except KeyboardInterrupt:
        logger.info("\n\nContinuous monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in continuous monitoring: {e}", exc_info=True)


def main():
    """Main entry point."""
    logger.info("GL-018 FLUEFLOW - Example Usage\n")

    # Run single analysis
    result = asyncio.run(run_flue_gas_analysis())

    # Uncomment to run continuous monitoring instead:
    # asyncio.run(continuous_monitoring_example())


if __name__ == "__main__":
    main()
