#!/usr/bin/env python3
"""
GL-017 CONDENSYNC - Example Usage
CondenserOptimizationAgent Comprehensive Usage Examples

This module provides comprehensive examples demonstrating the capabilities
of the GL-017 CONDENSYNC agent for condenser performance optimization.

Examples Include:
    1. Basic single condenser configuration
    2. Comprehensive SCADA integration setup
    3. Cooling tower coordination
    4. Multi-condenser plant configuration
    5. Performance optimization workflow
    6. Fouling analysis and cleaning recommendations
    7. Real-time monitoring loop
    8. Report generation

Author: GreenLang AI Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Example 1: Basic Single Condenser Configuration
# =============================================================================

def example_1_basic_configuration() -> Dict[str, Any]:
    """
    Example 1: Basic single condenser configuration for a 500 MW power plant.

    This example demonstrates how to configure a surface condenser with
    standard parameters for cooling water optimization.

    Returns:
        Configuration dictionary for a single condenser
    """
    logger.info("Example 1: Basic Single Condenser Configuration")

    config = {
        "agent_id": "GL-017",
        "condenser": {
            "id": "COND-001",
            "name": "Main Surface Condenser",
            "type": "SURFACE",
            "manufacturer": "GEA Heat Exchangers",
            "model": "NX-500",
            "commissioning_date": "2020-01-15",

            # Design specifications
            "design": {
                "heat_transfer_area_m2": 15000,
                "tube_count": 12000,
                "tube_od_mm": 25.4,
                "tube_id_mm": 22.9,
                "tube_length_m": 12.0,
                "tube_material": "TITANIUM",
                "tube_pattern": "TWO_PASS",
                "shell_diameter_m": 4.5,
                "design_pressure_kpa": 15,
                "design_vacuum_mmhg": 50,
                "design_duty_mw": 450,
                "design_u_value_w_m2k": 3500
            },

            # Operating limits
            "operating_limits": {
                "max_cw_inlet_temp_c": 35,
                "min_vacuum_mmhg": 25,
                "max_vacuum_mmhg": 100,
                "max_air_inleakage_kg_h": 15,
                "max_ttd_c": 5.0,
                "min_cleanliness_factor": 0.75,
                "max_condensate_subcooling_c": 3.0
            }
        },

        # Cooling water system
        "cooling_water": {
            "source": "RIVER",
            "system_type": "ONCE_THROUGH",
            "design_flow_rate_m3_h": 45000,
            "min_flow_rate_m3_h": 30000,
            "max_flow_rate_m3_h": 55000,
            "pump_count": 3,
            "pump_power_kw": 750,
            "vfd_equipped": True
        },

        # Vacuum system
        "vacuum_system": {
            "type": "SJAE",  # Steam Jet Air Ejector
            "stages": 2,
            "capacity_kg_h": 50,
            "motive_steam_pressure_bar": 4.0,
            "backup_vacuum_pump": True,
            "vacuum_pump_power_kw": 55
        }
    }

    logger.info(f"Configured condenser: {config['condenser']['name']}")
    logger.info(f"Design heat duty: {config['condenser']['design']['design_duty_mw']} MW")
    logger.info(f"Heat transfer area: {config['condenser']['design']['heat_transfer_area_m2']} m²")

    return config


# =============================================================================
# Example 2: Comprehensive SCADA Integration Setup
# =============================================================================

def example_2_scada_integration() -> Dict[str, Any]:
    """
    Example 2: Comprehensive SCADA integration with OPC-UA tag mapping.

    This example shows how to configure real-time data acquisition
    from plant SCADA systems for condenser monitoring.

    Returns:
        SCADA integration configuration
    """
    logger.info("Example 2: Comprehensive SCADA Integration Setup")

    scada_config = {
        "connection": {
            "protocol": "OPC-UA",
            "server_url": "opc.tcp://plant-scada.example.com:4840",
            "security_mode": "SignAndEncrypt",
            "security_policy": "Basic256Sha256",
            "certificate_path": "/etc/gl-017/certs/client.pem",
            "private_key_path": "/etc/gl-017/certs/client.key",
            "authentication": {
                "type": "certificate",
                "username": "gl-017-agent"
            },
            "timeout_ms": 5000,
            "retry_count": 3,
            "retry_delay_ms": 1000
        },

        # Tag mapping for condenser points
        "tag_mapping": {
            "cooling_water": {
                "CW_INLET_TEMP": {
                    "node_id": "ns=2;s=COND.CW.INLET.TEMP",
                    "description": "Cooling water inlet temperature",
                    "unit": "degC",
                    "range": {"min": 5, "max": 45},
                    "deadband": 0.1
                },
                "CW_OUTLET_TEMP": {
                    "node_id": "ns=2;s=COND.CW.OUTLET.TEMP",
                    "description": "Cooling water outlet temperature",
                    "unit": "degC",
                    "range": {"min": 10, "max": 50},
                    "deadband": 0.1
                },
                "CW_FLOW_RATE": {
                    "node_id": "ns=2;s=COND.CW.FLOW",
                    "description": "Cooling water flow rate",
                    "unit": "m3/h",
                    "range": {"min": 0, "max": 60000},
                    "deadband": 10
                },
                "CW_PRESSURE": {
                    "node_id": "ns=2;s=COND.CW.PRESSURE",
                    "description": "Cooling water inlet pressure",
                    "unit": "kPa",
                    "range": {"min": 0, "max": 500},
                    "deadband": 1
                }
            },

            "vacuum_system": {
                "VACUUM_PRESSURE": {
                    "node_id": "ns=2;s=COND.VACUUM.PRESSURE",
                    "description": "Condenser vacuum pressure",
                    "unit": "mmHg_abs",
                    "range": {"min": 20, "max": 200},
                    "deadband": 0.5
                },
                "HOTWELL_LEVEL": {
                    "node_id": "ns=2;s=COND.HOTWELL.LEVEL",
                    "description": "Hotwell level",
                    "unit": "percent",
                    "range": {"min": 0, "max": 100},
                    "deadband": 0.5
                },
                "AIR_EJECTOR_STEAM": {
                    "node_id": "ns=2;s=COND.SJAE.STEAM",
                    "description": "SJAE motive steam flow",
                    "unit": "kg/h",
                    "range": {"min": 0, "max": 1000},
                    "deadband": 5
                },
                "AIR_LEAKAGE_RATE": {
                    "node_id": "ns=2;s=COND.AIR.LEAKAGE",
                    "description": "Air in-leakage rate",
                    "unit": "kg/h",
                    "range": {"min": 0, "max": 100},
                    "deadband": 0.1
                }
            },

            "condensate": {
                "CONDENSATE_FLOW": {
                    "node_id": "ns=2;s=COND.COND.FLOW",
                    "description": "Condensate flow rate",
                    "unit": "kg/h",
                    "range": {"min": 0, "max": 2000000},
                    "deadband": 100
                },
                "CONDENSATE_TEMP": {
                    "node_id": "ns=2;s=COND.COND.TEMP",
                    "description": "Condensate temperature",
                    "unit": "degC",
                    "range": {"min": 20, "max": 60},
                    "deadband": 0.1
                }
            },

            "performance": {
                "TTD": {
                    "node_id": "ns=2;s=COND.PERF.TTD",
                    "description": "Terminal temperature difference",
                    "unit": "degC",
                    "range": {"min": 0, "max": 20},
                    "deadband": 0.05
                },
                "CLEANLINESS_FACTOR": {
                    "node_id": "ns=2;s=COND.PERF.CF",
                    "description": "Cleanliness factor",
                    "unit": "ratio",
                    "range": {"min": 0, "max": 1.2},
                    "deadband": 0.01
                },
                "HEAT_DUTY": {
                    "node_id": "ns=2;s=COND.PERF.DUTY",
                    "description": "Condenser heat duty",
                    "unit": "MW",
                    "range": {"min": 0, "max": 600},
                    "deadband": 1
                }
            }
        },

        # Setpoint tags for optimization output
        "setpoint_tags": {
            "CW_FLOW_SP": {
                "node_id": "ns=2;s=COND.CW.FLOW.SP",
                "description": "Cooling water flow setpoint",
                "unit": "m3/h",
                "write_enabled": True
            },
            "VACUUM_SP": {
                "node_id": "ns=2;s=COND.VACUUM.SP",
                "description": "Target vacuum pressure",
                "unit": "mmHg_abs",
                "write_enabled": True
            }
        },

        # Subscription configuration
        "subscription": {
            "publishing_interval_ms": 1000,
            "sampling_interval_ms": 500,
            "queue_size": 10,
            "discard_oldest": True
        }
    }

    logger.info(f"SCADA server: {scada_config['connection']['server_url']}")
    logger.info(f"Configured {len(scada_config['tag_mapping'])} tag groups")

    return scada_config


# =============================================================================
# Example 3: Cooling Tower Coordination
# =============================================================================

def example_3_cooling_tower_coordination() -> Dict[str, Any]:
    """
    Example 3: Cooling tower integration for optimized heat rejection.

    This example demonstrates coordination between the condenser
    and cooling tower systems for maximum efficiency.

    Returns:
        Cooling tower coordination configuration
    """
    logger.info("Example 3: Cooling Tower Coordination")

    cooling_tower_config = {
        "cooling_tower": {
            "id": "CT-001",
            "name": "Main Cooling Tower",
            "type": "MECHANICAL_DRAFT",
            "cells": 8,
            "design_capacity_mw": 500,
            "design_approach_c": 5.0,
            "design_range_c": 10.0,
            "design_wet_bulb_c": 24.0
        },

        # PLC communication
        "plc_connection": {
            "protocol": "MODBUS_TCP",
            "ip_address": "192.168.10.50",
            "port": 502,
            "unit_id": 1,
            "timeout_ms": 3000
        },

        # Register mapping
        "register_mapping": {
            "basin_temp": {"address": 40001, "type": "float", "scale": 0.1},
            "outlet_temp": {"address": 40003, "type": "float", "scale": 0.1},
            "fan_speeds": [
                {"address": 40010, "type": "uint16", "scale": 0.01, "cell": 1},
                {"address": 40011, "type": "uint16", "scale": 0.01, "cell": 2},
                {"address": 40012, "type": "uint16", "scale": 0.01, "cell": 3},
                {"address": 40013, "type": "uint16", "scale": 0.01, "cell": 4},
                {"address": 40014, "type": "uint16", "scale": 0.01, "cell": 5},
                {"address": 40015, "type": "uint16", "scale": 0.01, "cell": 6},
                {"address": 40016, "type": "uint16", "scale": 0.01, "cell": 7},
                {"address": 40017, "type": "uint16", "scale": 0.01, "cell": 8}
            ],
            "ambient_wet_bulb": {"address": 40020, "type": "float", "scale": 0.1},
            "ambient_dry_bulb": {"address": 40022, "type": "float", "scale": 0.1},
            "blowdown_rate": {"address": 40024, "type": "float", "scale": 0.01}
        },

        # Optimization parameters
        "optimization": {
            "mode": "AUTO",
            "target_approach_c": 5.0,
            "min_fan_speed_percent": 20,
            "max_fan_speed_percent": 100,
            "fan_staging": True,
            "vfd_equipped": True,
            "weather_compensation": True,
            "energy_optimization": True
        },

        # Cell management
        "cell_management": {
            "load_balancing": True,
            "runtime_equalization": True,
            "maintenance_stagger_days": 30,
            "min_active_cells": 4,
            "auto_isolation": True
        }
    }

    logger.info(f"Cooling tower: {cooling_tower_config['cooling_tower']['name']}")
    logger.info(f"Cells: {cooling_tower_config['cooling_tower']['cells']}")
    logger.info(f"Design capacity: {cooling_tower_config['cooling_tower']['design_capacity_mw']} MW")

    return cooling_tower_config


# =============================================================================
# Example 4: Multi-Condenser Plant Configuration
# =============================================================================

def example_4_multi_condenser_plant() -> Dict[str, Any]:
    """
    Example 4: Multi-condenser configuration for a combined cycle plant.

    This example shows how to configure multiple condensers
    for coordinated optimization in a large plant.

    Returns:
        Multi-condenser plant configuration
    """
    logger.info("Example 4: Multi-Condenser Plant Configuration")

    plant_config = {
        "plant": {
            "id": "PLANT-001",
            "name": "Riverside Combined Cycle Plant",
            "capacity_mw": 1200,
            "configuration": "2x2x1"  # 2 GTs, 2 HRSGs, 1 ST
        },

        "condensers": [
            {
                "id": "COND-LP-001",
                "name": "LP Condenser A",
                "turbine": "ST-001",
                "section": "LP",
                "heat_transfer_area_m2": 8000,
                "design_duty_mw": 200,
                "design_u_value_w_m2k": 3400,
                "priority": 1
            },
            {
                "id": "COND-LP-002",
                "name": "LP Condenser B",
                "turbine": "ST-001",
                "section": "LP",
                "heat_transfer_area_m2": 8000,
                "design_duty_mw": 200,
                "design_u_value_w_m2k": 3400,
                "priority": 2
            },
            {
                "id": "COND-IP-001",
                "name": "IP Condenser",
                "turbine": "ST-001",
                "section": "IP",
                "heat_transfer_area_m2": 4000,
                "design_duty_mw": 100,
                "design_u_value_w_m2k": 3200,
                "priority": 3
            }
        ],

        # Shared cooling water system
        "cooling_water_system": {
            "source": "COOLING_POND",
            "total_flow_capacity_m3_h": 80000,
            "header_pressure_kpa": 250,
            "return_header_pressure_kpa": 50,
            "pumps": [
                {"id": "CWP-001", "capacity_m3_h": 30000, "power_kw": 650},
                {"id": "CWP-002", "capacity_m3_h": 30000, "power_kw": 650},
                {"id": "CWP-003", "capacity_m3_h": 30000, "power_kw": 650}
            ]
        },

        # Coordinated optimization
        "coordination": {
            "mode": "PLANT_OPTIMIZED",
            "load_allocation": "EFFICIENCY_BASED",
            "flow_distribution": "DYNAMIC",
            "common_vacuum_target": True,
            "vacuum_margin_mmhg": 2.0
        },

        # Performance targets
        "targets": {
            "plant_vacuum_mmhg": 50,
            "max_ttd_c": 4.0,
            "min_cleanliness_factor": 0.80,
            "max_backpressure_deviation_percent": 3.0
        }
    }

    logger.info(f"Plant: {plant_config['plant']['name']}")
    logger.info(f"Total condensers: {len(plant_config['condensers'])}")
    total_duty = sum(c['design_duty_mw'] for c in plant_config['condensers'])
    logger.info(f"Total design duty: {total_duty} MW")

    return plant_config


# =============================================================================
# Example 5: Performance Optimization Workflow
# =============================================================================

async def example_5_performance_optimization() -> Dict[str, Any]:
    """
    Example 5: Complete performance optimization workflow.

    This example demonstrates the full optimization cycle
    including data collection, analysis, and recommendations.

    Returns:
        Optimization workflow results
    """
    logger.info("Example 5: Performance Optimization Workflow")

    # Simulated real-time data (would come from SCADA in production)
    current_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "cooling_water": {
            "inlet_temp_c": 28.5,
            "outlet_temp_c": 38.2,
            "flow_rate_m3_h": 42000,
            "pressure_kpa": 180
        },
        "vacuum": {
            "pressure_mmhg": 52.5,
            "saturation_temp_c": 39.8,
            "air_leakage_kg_h": 8.5
        },
        "condensate": {
            "flow_rate_kg_h": 850000,
            "temperature_c": 38.5,
            "subcooling_c": 1.3
        },
        "turbine": {
            "load_mw": 480,
            "exhaust_temp_c": 40.2,
            "exhaust_pressure_kpa": 7.0
        }
    }

    logger.info("Step 1: Collecting real-time data...")
    logger.info(f"  CW Inlet: {current_data['cooling_water']['inlet_temp_c']}°C")
    logger.info(f"  CW Outlet: {current_data['cooling_water']['outlet_temp_c']}°C")
    logger.info(f"  Vacuum: {current_data['vacuum']['pressure_mmhg']} mmHg abs")

    # Step 2: Calculate performance metrics
    logger.info("Step 2: Calculating performance metrics...")

    cw_temp_rise = (current_data['cooling_water']['outlet_temp_c'] -
                   current_data['cooling_water']['inlet_temp_c'])

    # Heat duty calculation: Q = m * Cp * ΔT
    cp_water = 4.186  # kJ/kg·K
    water_density = 1000  # kg/m³
    mass_flow_kg_s = (current_data['cooling_water']['flow_rate_m3_h'] *
                     water_density / 3600)
    heat_duty_mw = mass_flow_kg_s * cp_water * cw_temp_rise / 1000

    # TTD calculation
    ttd = (current_data['vacuum']['saturation_temp_c'] -
           current_data['cooling_water']['outlet_temp_c'])

    # LMTD calculation
    delta_t1 = (current_data['vacuum']['saturation_temp_c'] -
               current_data['cooling_water']['inlet_temp_c'])
    delta_t2 = (current_data['vacuum']['saturation_temp_c'] -
               current_data['cooling_water']['outlet_temp_c'])

    import math
    if delta_t1 > 0 and delta_t2 > 0 and delta_t1 != delta_t2:
        lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)
    else:
        lmtd = (delta_t1 + delta_t2) / 2

    # U-value calculation (assuming 15000 m² area)
    heat_transfer_area = 15000
    u_value = (heat_duty_mw * 1e6) / (heat_transfer_area * lmtd)

    # Cleanliness factor (assuming design U = 3500)
    design_u = 3500
    cleanliness_factor = u_value / design_u

    performance_metrics = {
        "heat_duty_mw": round(heat_duty_mw, 2),
        "ttd_c": round(ttd, 2),
        "lmtd_c": round(lmtd, 2),
        "u_value_w_m2k": round(u_value, 1),
        "cleanliness_factor": round(cleanliness_factor, 3),
        "cw_temp_rise_c": round(cw_temp_rise, 2)
    }

    logger.info(f"  Heat Duty: {performance_metrics['heat_duty_mw']} MW")
    logger.info(f"  TTD: {performance_metrics['ttd_c']}°C")
    logger.info(f"  U-value: {performance_metrics['u_value_w_m2k']} W/m²·K")
    logger.info(f"  Cleanliness: {performance_metrics['cleanliness_factor']}")

    # Step 3: Generate optimization recommendations
    logger.info("Step 3: Generating optimization recommendations...")

    recommendations = []

    # Check vacuum
    optimal_vacuum = 48.0  # mmHg abs
    if current_data['vacuum']['pressure_mmhg'] > optimal_vacuum + 3:
        vacuum_improvement = current_data['vacuum']['pressure_mmhg'] - optimal_vacuum
        heat_rate_improvement = vacuum_improvement * 0.15  # ~0.15% per mmHg
        recommendations.append({
            "type": "VACUUM_OPTIMIZATION",
            "priority": "HIGH",
            "current_value": current_data['vacuum']['pressure_mmhg'],
            "target_value": optimal_vacuum,
            "expected_benefit": f"{heat_rate_improvement:.1f}% heat rate improvement",
            "action": "Increase cooling water flow or check air in-leakage"
        })

    # Check cleanliness
    if cleanliness_factor < 0.85:
        recommendations.append({
            "type": "TUBE_CLEANING",
            "priority": "MEDIUM",
            "current_value": cleanliness_factor,
            "target_value": 0.95,
            "expected_benefit": f"Restore U-value to {design_u * 0.95:.0f} W/m²·K",
            "action": "Schedule tube cleaning within 30 days"
        })

    # Check air in-leakage
    if current_data['vacuum']['air_leakage_kg_h'] > 10:
        recommendations.append({
            "type": "AIR_LEAKAGE",
            "priority": "HIGH",
            "current_value": current_data['vacuum']['air_leakage_kg_h'],
            "target_value": 5.0,
            "expected_benefit": "Improved vacuum and reduced SJAE steam consumption",
            "action": "Perform leak detection survey"
        })

    # Check subcooling
    if current_data['condensate']['subcooling_c'] > 2.0:
        recommendations.append({
            "type": "SUBCOOLING",
            "priority": "LOW",
            "current_value": current_data['condensate']['subcooling_c'],
            "target_value": 1.0,
            "expected_benefit": "Improved heat recovery efficiency",
            "action": "Check hotwell level and venting"
        })

    logger.info(f"  Generated {len(recommendations)} recommendations")
    for rec in recommendations:
        logger.info(f"    - [{rec['priority']}] {rec['type']}: {rec['action']}")

    # Step 4: Calculate potential savings
    logger.info("Step 4: Calculating potential savings...")

    # Estimated savings based on optimizations
    baseline_heat_rate_kj_kwh = 7500
    turbine_output_kw = current_data['turbine']['load_mw'] * 1000
    operating_hours_per_year = 8000
    fuel_cost_per_gj = 5.0

    total_heat_rate_improvement = 0
    for rec in recommendations:
        if rec['type'] == 'VACUUM_OPTIMIZATION':
            total_heat_rate_improvement += float(rec['expected_benefit'].split('%')[0])
        elif rec['type'] == 'TUBE_CLEANING':
            total_heat_rate_improvement += 0.5  # ~0.5% improvement

    fuel_savings_gj_year = (turbine_output_kw * operating_hours_per_year *
                           baseline_heat_rate_kj_kwh *
                           (total_heat_rate_improvement / 100) / 1e6)
    cost_savings_year = fuel_savings_gj_year * fuel_cost_per_gj

    savings = {
        "heat_rate_improvement_percent": round(total_heat_rate_improvement, 2),
        "fuel_savings_gj_year": round(fuel_savings_gj_year, 0),
        "cost_savings_usd_year": round(cost_savings_year, 0)
    }

    logger.info(f"  Heat Rate Improvement: {savings['heat_rate_improvement_percent']}%")
    logger.info(f"  Annual Fuel Savings: {savings['fuel_savings_gj_year']} GJ")
    logger.info(f"  Annual Cost Savings: ${savings['cost_savings_usd_year']:,.0f}")

    # Compile results
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "SUCCESS",
        "current_data": current_data,
        "performance_metrics": performance_metrics,
        "recommendations": recommendations,
        "potential_savings": savings
    }

    return results


# =============================================================================
# Example 6: Fouling Analysis and Cleaning Recommendations
# =============================================================================

async def example_6_fouling_analysis() -> Dict[str, Any]:
    """
    Example 6: Comprehensive fouling analysis workflow.

    This example demonstrates tube fouling prediction and
    optimal cleaning schedule determination.

    Returns:
        Fouling analysis results with cleaning recommendations
    """
    logger.info("Example 6: Fouling Analysis and Cleaning Recommendations")

    # Historical performance data (simulated)
    historical_data = [
        {"date": "2024-01-01", "cleanliness_factor": 0.98, "u_value": 3430},
        {"date": "2024-02-01", "cleanliness_factor": 0.95, "u_value": 3325},
        {"date": "2024-03-01", "cleanliness_factor": 0.91, "u_value": 3185},
        {"date": "2024-04-01", "cleanliness_factor": 0.88, "u_value": 3080},
        {"date": "2024-05-01", "cleanliness_factor": 0.85, "u_value": 2975},
        {"date": "2024-06-01", "cleanliness_factor": 0.82, "u_value": 2870},
    ]

    logger.info("Step 1: Analyzing historical fouling trend...")

    # Calculate fouling rate (cleanliness factor decline per month)
    cf_values = [d['cleanliness_factor'] for d in historical_data]
    fouling_rate_per_month = (cf_values[0] - cf_values[-1]) / (len(cf_values) - 1)

    current_cf = cf_values[-1]
    design_u = 3500
    current_u = historical_data[-1]['u_value']

    logger.info(f"  Current cleanliness factor: {current_cf}")
    logger.info(f"  Current U-value: {current_u} W/m²·K")
    logger.info(f"  Fouling rate: {fouling_rate_per_month:.3f} CF/month")

    # Step 2: Predict future performance
    logger.info("Step 2: Predicting future performance...")

    predictions = []
    min_acceptable_cf = 0.75

    for months_ahead in range(1, 7):
        predicted_cf = current_cf - (fouling_rate_per_month * months_ahead)
        predicted_u = design_u * predicted_cf
        predictions.append({
            "months_ahead": months_ahead,
            "predicted_cf": round(predicted_cf, 3),
            "predicted_u": round(predicted_u, 0),
            "acceptable": predicted_cf >= min_acceptable_cf
        })

    # Find when cleaning is required
    months_until_cleaning = None
    for pred in predictions:
        if not pred['acceptable']:
            months_until_cleaning = pred['months_ahead']
            break

    logger.info(f"  Predicted to reach minimum CF in: {months_until_cleaning} months")

    # Step 3: Calculate cleaning economics
    logger.info("Step 3: Calculating cleaning economics...")

    cleaning_cost = 50000  # USD for tube cleaning
    heat_rate_penalty_per_cf = 0.5  # % per 0.01 CF
    turbine_output_mw = 500
    operating_hours_month = 720
    fuel_cost_per_mwh = 30

    # Cost of delayed cleaning (running with low CF)
    cf_deficit = 0.95 - current_cf  # Compared to post-cleaning target
    heat_rate_penalty = cf_deficit * 100 * heat_rate_penalty_per_cf

    fuel_penalty_per_month = (turbine_output_mw * operating_hours_month *
                              fuel_cost_per_mwh * (heat_rate_penalty / 100))

    economics = {
        "cleaning_cost_usd": cleaning_cost,
        "current_cf_penalty_percent": round(heat_rate_penalty, 2),
        "fuel_penalty_usd_per_month": round(fuel_penalty_per_month, 0),
        "payback_months": round(cleaning_cost / fuel_penalty_per_month, 1) if fuel_penalty_per_month > 0 else 0
    }

    logger.info(f"  Cleaning cost: ${economics['cleaning_cost_usd']:,}")
    logger.info(f"  Monthly fuel penalty: ${economics['fuel_penalty_usd_per_month']:,}")
    logger.info(f"  Payback period: {economics['payback_months']} months")

    # Step 4: Generate cleaning recommendation
    logger.info("Step 4: Generating cleaning recommendation...")

    recommendation = {
        "recommended_action": "SCHEDULE_CLEANING",
        "urgency": "MEDIUM" if months_until_cleaning and months_until_cleaning > 2 else "HIGH",
        "optimal_cleaning_date": None,
        "cleaning_method": "MECHANICAL_BRUSH",
        "expected_cf_after_cleaning": 0.95,
        "expected_u_after_cleaning": design_u * 0.95,
        "rationale": []
    }

    # Determine optimal date
    import datetime as dt
    if months_until_cleaning:
        # Clean 1 month before reaching minimum
        optimal_months = max(1, months_until_cleaning - 1)
        optimal_date = datetime.utcnow() + timedelta(days=optimal_months * 30)
        recommendation["optimal_cleaning_date"] = optimal_date.strftime("%Y-%m-%d")

    # Add rationale
    recommendation["rationale"].append(
        f"Current CF ({current_cf}) below optimal (0.90)"
    )
    recommendation["rationale"].append(
        f"Fouling rate of {fouling_rate_per_month:.3f}/month indicates active degradation"
    )
    recommendation["rationale"].append(
        f"Economic analysis shows {economics['payback_months']}-month payback"
    )

    logger.info(f"  Recommendation: {recommendation['recommended_action']}")
    logger.info(f"  Urgency: {recommendation['urgency']}")
    logger.info(f"  Optimal date: {recommendation['optimal_cleaning_date']}")

    # Compile results
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "current_status": {
            "cleanliness_factor": current_cf,
            "u_value_w_m2k": current_u,
            "fouling_rate_per_month": round(fouling_rate_per_month, 4)
        },
        "predictions": predictions,
        "economics": economics,
        "recommendation": recommendation
    }

    return results


# =============================================================================
# Example 7: Real-Time Monitoring Loop
# =============================================================================

async def example_7_monitoring_loop(duration_seconds: int = 30) -> None:
    """
    Example 7: Real-time monitoring loop simulation.

    This example demonstrates continuous condenser monitoring
    with alerting capabilities.

    Args:
        duration_seconds: Duration to run the monitoring loop
    """
    logger.info("Example 7: Real-Time Monitoring Loop")
    logger.info(f"Running for {duration_seconds} seconds...")

    import random

    # Alert thresholds
    thresholds = {
        "vacuum_high_mmhg": 60,
        "vacuum_critical_mmhg": 70,
        "ttd_high_c": 4.0,
        "ttd_critical_c": 6.0,
        "air_leakage_high_kg_h": 10,
        "air_leakage_critical_kg_h": 20,
        "cf_low": 0.80,
        "cf_critical": 0.70
    }

    alerts_triggered = []
    cycle_count = 0
    start_time = datetime.utcnow()

    # Monitoring loop
    while (datetime.utcnow() - start_time).total_seconds() < duration_seconds:
        cycle_count += 1

        # Simulate sensor readings with some variation
        readings = {
            "vacuum_mmhg": 52 + random.uniform(-3, 8),
            "ttd_c": 3.5 + random.uniform(-0.5, 1.5),
            "air_leakage_kg_h": 7 + random.uniform(-2, 8),
            "cleanliness_factor": 0.85 + random.uniform(-0.05, 0.05),
            "cw_inlet_c": 28 + random.uniform(-1, 1),
            "cw_outlet_c": 38 + random.uniform(-1, 1)
        }

        # Check for alerts
        new_alerts = []

        if readings["vacuum_mmhg"] > thresholds["vacuum_critical_mmhg"]:
            new_alerts.append({
                "severity": "CRITICAL",
                "type": "HIGH_VACUUM",
                "value": readings["vacuum_mmhg"],
                "threshold": thresholds["vacuum_critical_mmhg"],
                "message": f"Critical vacuum: {readings['vacuum_mmhg']:.1f} mmHg"
            })
        elif readings["vacuum_mmhg"] > thresholds["vacuum_high_mmhg"]:
            new_alerts.append({
                "severity": "WARNING",
                "type": "HIGH_VACUUM",
                "value": readings["vacuum_mmhg"],
                "threshold": thresholds["vacuum_high_mmhg"],
                "message": f"High vacuum: {readings['vacuum_mmhg']:.1f} mmHg"
            })

        if readings["air_leakage_kg_h"] > thresholds["air_leakage_critical_kg_h"]:
            new_alerts.append({
                "severity": "CRITICAL",
                "type": "AIR_LEAKAGE",
                "value": readings["air_leakage_kg_h"],
                "threshold": thresholds["air_leakage_critical_kg_h"],
                "message": f"Critical air leakage: {readings['air_leakage_kg_h']:.1f} kg/h"
            })

        if readings["cleanliness_factor"] < thresholds["cf_critical"]:
            new_alerts.append({
                "severity": "CRITICAL",
                "type": "LOW_CLEANLINESS",
                "value": readings["cleanliness_factor"],
                "threshold": thresholds["cf_critical"],
                "message": f"Critical cleanliness: {readings['cleanliness_factor']:.2f}"
            })

        # Log status
        if cycle_count % 5 == 0:  # Log every 5 cycles
            logger.info(
                f"Cycle {cycle_count}: Vacuum={readings['vacuum_mmhg']:.1f}mmHg, "
                f"TTD={readings['ttd_c']:.2f}°C, CF={readings['cleanliness_factor']:.2f}"
            )

        # Process alerts
        for alert in new_alerts:
            alerts_triggered.append({
                "timestamp": datetime.utcnow().isoformat(),
                **alert
            })
            logger.warning(f"ALERT [{alert['severity']}]: {alert['message']}")

        # Wait for next cycle
        await asyncio.sleep(1)

    # Summary
    logger.info(f"\nMonitoring Summary:")
    logger.info(f"  Total cycles: {cycle_count}")
    logger.info(f"  Total alerts: {len(alerts_triggered)}")

    critical_count = sum(1 for a in alerts_triggered if a['severity'] == 'CRITICAL')
    warning_count = sum(1 for a in alerts_triggered if a['severity'] == 'WARNING')
    logger.info(f"  Critical alerts: {critical_count}")
    logger.info(f"  Warning alerts: {warning_count}")


# =============================================================================
# Example 8: Performance Report Generation
# =============================================================================

async def example_8_generate_report() -> Dict[str, Any]:
    """
    Example 8: Comprehensive performance report generation.

    This example demonstrates how to generate a detailed
    condenser performance report for management review.

    Returns:
        Complete performance report data
    """
    logger.info("Example 8: Performance Report Generation")

    report_period = {
        "start_date": "2024-06-01",
        "end_date": "2024-06-30",
        "report_type": "MONTHLY",
        "condenser_id": "COND-001"
    }

    # Simulated monthly statistics
    monthly_stats = {
        "operating_hours": 720,
        "availability_percent": 99.5,
        "avg_vacuum_mmhg": 51.2,
        "min_vacuum_mmhg": 48.5,
        "max_vacuum_mmhg": 58.3,
        "avg_ttd_c": 3.8,
        "avg_cleanliness_factor": 0.84,
        "avg_heat_duty_mw": 425,
        "total_heat_rejected_gwh": 306,
        "avg_cw_flow_m3_h": 43500,
        "avg_air_leakage_kg_h": 7.2
    }

    # Performance vs targets
    targets = {
        "vacuum_mmhg": {"target": 50, "actual": monthly_stats["avg_vacuum_mmhg"]},
        "ttd_c": {"target": 3.5, "actual": monthly_stats["avg_ttd_c"]},
        "cleanliness_factor": {"target": 0.85, "actual": monthly_stats["avg_cleanliness_factor"]},
        "availability_percent": {"target": 99.0, "actual": monthly_stats["availability_percent"]}
    }

    for key, val in targets.items():
        if "vacuum" in key or "ttd" in key:
            val["status"] = "GOOD" if val["actual"] <= val["target"] * 1.05 else "NEEDS_ATTENTION"
        else:
            val["status"] = "GOOD" if val["actual"] >= val["target"] * 0.95 else "NEEDS_ATTENTION"

    # Calculated KPIs
    design_u = 3500
    actual_u = design_u * monthly_stats["avg_cleanliness_factor"]
    u_value_loss = design_u - actual_u

    # Heat rate impact
    vacuum_deviation = monthly_stats["avg_vacuum_mmhg"] - 50
    heat_rate_impact_percent = vacuum_deviation * 0.15  # 0.15% per mmHg

    # Cost impact (simplified)
    turbine_output_mw = 500
    fuel_cost_per_mwh = 30
    heat_rate_cost_impact = (turbine_output_mw * monthly_stats["operating_hours"] *
                            fuel_cost_per_mwh * heat_rate_impact_percent / 100)

    kpis = {
        "condenser_performance_index": round(monthly_stats["avg_cleanliness_factor"] * 100, 1),
        "u_value_actual_w_m2k": round(actual_u, 0),
        "u_value_loss_w_m2k": round(u_value_loss, 0),
        "heat_rate_impact_percent": round(heat_rate_impact_percent, 2),
        "cost_impact_usd": round(heat_rate_cost_impact, 0)
    }

    # Events and incidents
    events = [
        {"date": "2024-06-05", "type": "ALARM", "description": "High vacuum alarm - 62 mmHg for 15 min"},
        {"date": "2024-06-12", "type": "MAINTENANCE", "description": "SJAE inspection completed"},
        {"date": "2024-06-18", "type": "ALARM", "description": "Air leakage alarm - leak detected on LP gland"},
        {"date": "2024-06-20", "type": "REPAIR", "description": "LP turbine gland seal replaced"},
    ]

    # Recommendations
    recommendations = [
        {
            "priority": "HIGH",
            "recommendation": "Schedule tube cleaning within 45 days",
            "rationale": "Cleanliness factor trending below target",
            "estimated_benefit": "0.5% heat rate improvement"
        },
        {
            "priority": "MEDIUM",
            "recommendation": "Perform comprehensive air leak survey",
            "rationale": "Air leakage above optimal levels",
            "estimated_benefit": "Reduced SJAE steam consumption"
        },
        {
            "priority": "LOW",
            "recommendation": "Review cooling tower approach temperature",
            "rationale": "Opportunity for CW inlet temperature reduction",
            "estimated_benefit": "1-2 mmHg vacuum improvement"
        }
    ]

    # Compile report
    report = {
        "report_metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "generated_by": "GL-017 CONDENSYNC",
            "version": "1.0.0",
            **report_period
        },
        "executive_summary": {
            "overall_status": "SATISFACTORY",
            "key_findings": [
                f"Condenser availability: {monthly_stats['availability_percent']}% (Target: 99%)",
                f"Average vacuum: {monthly_stats['avg_vacuum_mmhg']} mmHg (Target: 50 mmHg)",
                f"Cleanliness factor: {monthly_stats['avg_cleanliness_factor']} (Target: 0.85)",
                f"Heat rate impact: {kpis['heat_rate_impact_percent']}% penalty"
            ],
            "action_items": len([r for r in recommendations if r['priority'] == 'HIGH'])
        },
        "monthly_statistics": monthly_stats,
        "performance_vs_targets": targets,
        "key_performance_indicators": kpis,
        "events_and_incidents": events,
        "recommendations": recommendations
    }

    logger.info("Report generated successfully")
    logger.info(f"  Period: {report_period['start_date']} to {report_period['end_date']}")
    logger.info(f"  Overall Status: {report['executive_summary']['overall_status']}")
    logger.info(f"  High-priority recommendations: {report['executive_summary']['action_items']}")

    return report


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """
    Main entry point demonstrating all example workflows.
    """
    print("=" * 70)
    print("GL-017 CONDENSYNC - CondenserOptimizationAgent")
    print("Comprehensive Usage Examples")
    print("=" * 70)
    print()

    # Example 1: Basic Configuration
    print("-" * 70)
    config = example_1_basic_configuration()
    print()

    # Example 2: SCADA Integration
    print("-" * 70)
    scada_config = example_2_scada_integration()
    print()

    # Example 3: Cooling Tower Coordination
    print("-" * 70)
    ct_config = example_3_cooling_tower_coordination()
    print()

    # Example 4: Multi-Condenser Plant
    print("-" * 70)
    plant_config = example_4_multi_condenser_plant()
    print()

    # Example 5: Performance Optimization
    print("-" * 70)
    optimization_results = await example_5_performance_optimization()
    print()

    # Example 6: Fouling Analysis
    print("-" * 70)
    fouling_results = await example_6_fouling_analysis()
    print()

    # Example 7: Real-Time Monitoring (short demo)
    print("-" * 70)
    await example_7_monitoring_loop(duration_seconds=10)
    print()

    # Example 8: Report Generation
    print("-" * 70)
    report = await example_8_generate_report()
    print()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)

    return {
        "config": config,
        "scada_config": scada_config,
        "cooling_tower_config": ct_config,
        "plant_config": plant_config,
        "optimization_results": optimization_results,
        "fouling_results": fouling_results,
        "report": report
    }


if __name__ == "__main__":
    asyncio.run(main())
