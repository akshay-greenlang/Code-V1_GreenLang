"""
Deterministic tool functions for ProcessHeatOrchestrator.

This module implements all deterministic calculation and optimization functions
for process heat operations. All functions follow zero-hallucination principles
with no LLM involvement in calculations.
"""

import hashlib
import logging
import math
import threading
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThermalEfficiencyResult:
    """Result of thermal efficiency calculation."""
    overall_efficiency: float
    carnot_efficiency: float
    heat_recovery_efficiency: float
    losses: Dict[str, float]
    timestamp: str
    provenance_hash: str


@dataclass
class HeatDistributionStrategy:
    """Optimized heat distribution strategy."""
    distribution_matrix: Dict[str, Dict[str, float]]
    total_heat_demand_mw: float
    total_heat_supply_mw: float
    optimization_score: float
    constraints_satisfied: bool
    timestamp: str
    provenance_hash: str


@dataclass
class EnergyBalance:
    """Energy balance validation result."""
    input_energy_mw: float
    output_energy_mw: float
    losses_mw: float
    balance_error_percent: float
    is_valid: bool
    violations: List[str]
    timestamp: str
    provenance_hash: str


@dataclass
class ComplianceResult:
    """Emissions compliance check result."""
    total_emissions_kg_hr: float
    emission_intensity_kg_mwh: float
    regulatory_limit_kg_mwh: float
    compliance_status: str  # PASS, FAIL, WARNING
    margin_percent: float
    violations: List[str]
    timestamp: str
    provenance_hash: str


class ProcessHeatTools:
    """
    Deterministic tool functions for process heat operations.

    All methods implement zero-hallucination calculations using
    deterministic algorithms and formulas only.
    """

    @staticmethod
    def calculate_thermal_efficiency(plant_data: Dict[str, Any]) -> ThermalEfficiencyResult:  # type: ignore
        """
        Calculate thermal efficiency for process heat system.

        Args:
            plant_data: Dictionary containing plant operating data
                - inlet_temp_c: Inlet temperature in Celsius
                - outlet_temp_c: Outlet temperature in Celsius
                - ambient_temp_c: Ambient temperature in Celsius
                - fuel_input_mw: Fuel energy input in MW
                - useful_heat_mw: Useful heat output in MW
                - heat_recovery_mw: Heat recovered in MW

        Returns:
            ThermalEfficiencyResult with efficiency metrics
        """
        try:
            # Extract data
            inlet_temp_k = plant_data.get('inlet_temp_c', 500) + 273.15
            outlet_temp_k = plant_data.get('outlet_temp_c', 150) + 273.15
            ambient_temp_k = plant_data.get('ambient_temp_c', 25) + 273.15
            fuel_input = plant_data.get('fuel_input_mw', 100)
            useful_heat = plant_data.get('useful_heat_mw', 85)
            heat_recovery = plant_data.get('heat_recovery_mw', 0)

            # Calculate Carnot efficiency (theoretical maximum)
            carnot_efficiency = 1 - (ambient_temp_k / inlet_temp_k)
            carnot_efficiency = max(0, min(1, carnot_efficiency))  # Clamp to [0, 1]

            # Calculate actual thermal efficiency
            total_useful_output = useful_heat + heat_recovery
            overall_efficiency = total_useful_output / fuel_input if fuel_input > 0 else 0
            overall_efficiency = max(0, min(1, overall_efficiency))  # Clamp to [0, 1]

            # Calculate heat recovery efficiency
            waste_heat = fuel_input - useful_heat
            heat_recovery_efficiency = heat_recovery / waste_heat if waste_heat > 0 else 0
            heat_recovery_efficiency = max(0, min(1, heat_recovery_efficiency))

            # Calculate losses
            losses = {
                'flue_gas': (fuel_input - total_useful_output) * 0.5,  # 50% of losses to flue gas
                'radiation': (fuel_input - total_useful_output) * 0.2,  # 20% radiation losses
                'convection': (fuel_input - total_useful_output) * 0.2,  # 20% convection losses
                'other': (fuel_input - total_useful_output) * 0.1       # 10% other losses
            }

            timestamp = datetime.utcnow().isoformat()

            # Calculate provenance hash
            provenance_str = f"{plant_data}{overall_efficiency}{carnot_efficiency}{timestamp}"
            provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

            return ThermalEfficiencyResult(
                overall_efficiency=round(overall_efficiency * 100, 2),
                carnot_efficiency=round(carnot_efficiency * 100, 2),
                heat_recovery_efficiency=round(heat_recovery_efficiency * 100, 2),
                losses=losses,
                timestamp=timestamp,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            logger.error(f"Thermal efficiency calculation failed: {str(e)}")
            raise

    @staticmethod
    def optimize_heat_distribution(
        sensor_feeds: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> HeatDistributionStrategy:  # type: ignore
        """
        Optimize heat distribution across process units.

        Uses linear programming principles for deterministic optimization.

        Args:
            sensor_feeds: Real-time sensor data from process units
            constraints: Operational constraints and limits

        Returns:
            Optimized heat distribution strategy
        """
        try:
            # Extract demand and supply data
            heat_demands = sensor_feeds.get('heat_demands', {})  # MW per unit
            heat_sources = sensor_feeds.get('heat_sources', {})  # MW available
            temperatures = sensor_feeds.get('temperatures', {})  # Current temps

            # Extract constraints
            max_temp_variance = constraints.get('max_temperature_variance_c', 5.0)
            min_efficiency = constraints.get('min_efficiency_percent', 85.0)
            priority_units = constraints.get('priority_units', [])

            # Initialize distribution matrix
            distribution_matrix = {}

            # Calculate total supply and demand
            total_demand = sum(heat_demands.values())
            total_supply = sum(heat_sources.values())

            # Simple proportional distribution with priority handling
            remaining_supply = total_supply

            # First, allocate to priority units
            for unit in priority_units:
                if unit in heat_demands:
                    allocation = min(heat_demands[unit], remaining_supply)
                    distribution_matrix[unit] = {
                        'allocated_mw': allocation,
                        'demand_mw': heat_demands[unit],
                        'satisfaction_percent': (allocation / heat_demands[unit] * 100) if heat_demands[unit] > 0 else 100
                    }
                    remaining_supply -= allocation

            # Then allocate to remaining units proportionally
            non_priority_units = [u for u in heat_demands if u not in priority_units]
            non_priority_demand = sum(heat_demands[u] for u in non_priority_units)

            for unit in non_priority_units:
                if non_priority_demand > 0:
                    proportion = heat_demands[unit] / non_priority_demand
                    allocation = min(heat_demands[unit], remaining_supply * proportion)
                else:
                    allocation = 0

                distribution_matrix[unit] = {
                    'allocated_mw': allocation,
                    'demand_mw': heat_demands[unit],
                    'satisfaction_percent': (allocation / heat_demands[unit] * 100) if heat_demands[unit] > 0 else 100
                }

            # Calculate optimization score (0-100)
            avg_satisfaction = np.mean([d['satisfaction_percent'] for d in distribution_matrix.values()])
            efficiency_score = min(100, (total_supply / total_demand * 100)) if total_demand > 0 else 100
            optimization_score = (avg_satisfaction * 0.7 + efficiency_score * 0.3)

            # Check constraints
            constraints_satisfied = (
                avg_satisfaction >= min_efficiency and
                total_supply >= total_demand * 0.95  # At least 95% of demand met
            )

            timestamp = datetime.utcnow().isoformat()

            # Calculate provenance hash
            provenance_str = f"{sensor_feeds}{constraints}{distribution_matrix}{timestamp}"
            provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

            return HeatDistributionStrategy(
                distribution_matrix=distribution_matrix,
                total_heat_demand_mw=round(total_demand, 2),
                total_heat_supply_mw=round(total_supply, 2),
                optimization_score=round(optimization_score, 2),
                constraints_satisfied=constraints_satisfied,
                timestamp=timestamp,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            logger.error(f"Heat distribution optimization failed: {str(e)}")
            raise

    @staticmethod
    def validate_energy_balance(consumption_data: Dict[str, Any]) -> EnergyBalance:  # type: ignore
        """
        Validate energy balance across the system.

        Args:
            consumption_data: Energy consumption and generation data

        Returns:
            Energy balance validation result
        """
        try:
            # Extract energy flows
            fuel_input = consumption_data.get('fuel_input_mw', 0)
            electricity_input = consumption_data.get('electricity_input_mw', 0)
            steam_import = consumption_data.get('steam_import_mw', 0)

            process_heat_output = consumption_data.get('process_heat_output_mw', 0)
            electricity_output = consumption_data.get('electricity_output_mw', 0)
            steam_export = consumption_data.get('steam_export_mw', 0)

            measured_losses = consumption_data.get('measured_losses_mw', 0)

            # Calculate totals
            total_input = fuel_input + electricity_input + steam_import
            total_output = process_heat_output + electricity_output + steam_export
            calculated_losses = total_input - total_output

            # Validate energy balance (should be close to zero after accounting for losses)
            balance_error = abs(calculated_losses - measured_losses)
            balance_error_percent = (balance_error / total_input * 100) if total_input > 0 else 0

            # Tolerance for balance validation (2% typically acceptable)
            tolerance_percent = 2.0
            is_valid = balance_error_percent <= tolerance_percent

            # Identify violations
            violations = []
            if balance_error_percent > tolerance_percent:
                violations.append(f"Balance error {balance_error_percent:.2f}% exceeds tolerance {tolerance_percent}%")
            if calculated_losses < 0:
                violations.append("Negative losses calculated - output exceeds input")
            if total_input == 0:
                violations.append("No energy input detected")

            timestamp = datetime.utcnow().isoformat()

            # Calculate provenance hash
            provenance_str = f"{consumption_data}{total_input}{total_output}{timestamp}"
            provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

            return EnergyBalance(
                input_energy_mw=round(total_input, 2),
                output_energy_mw=round(total_output, 2),
                losses_mw=round(calculated_losses, 2),
                balance_error_percent=round(balance_error_percent, 2),
                is_valid=is_valid,
                violations=violations,
                timestamp=timestamp,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            logger.error(f"Energy balance validation failed: {str(e)}")
            raise

    @staticmethod
    def check_emissions_compliance(
        emissions_data: Dict[str, Any],
        regulations: Dict[str, Any]
    ) -> ComplianceResult:  # type: ignore
        """
        Check emissions compliance against regulations.

        Args:
            emissions_data: Current emissions measurements
            regulations: Applicable emission regulations

        Returns:
            Compliance check result
        """
        try:
            # Extract emissions data
            co2_kg_hr = emissions_data.get('co2_kg_hr', 0)
            nox_kg_hr = emissions_data.get('nox_kg_hr', 0)
            sox_kg_hr = emissions_data.get('sox_kg_hr', 0)
            particulate_kg_hr = emissions_data.get('particulate_kg_hr', 0)
            heat_output_mw = emissions_data.get('heat_output_mw', 100)

            # Calculate total emissions
            total_emissions = co2_kg_hr + nox_kg_hr + sox_kg_hr + particulate_kg_hr

            # Calculate emission intensity
            emission_intensity = (total_emissions / heat_output_mw) if heat_output_mw > 0 else 0

            # Get regulatory limit
            regulatory_limit = regulations.get('max_emissions_kg_mwh', 200)
            warning_threshold = regulatory_limit * 0.9  # 90% of limit triggers warning

            # Determine compliance status
            if emission_intensity <= warning_threshold:
                compliance_status = "PASS"
            elif emission_intensity <= regulatory_limit:
                compliance_status = "WARNING"
            else:
                compliance_status = "FAIL"

            # Calculate margin
            margin = regulatory_limit - emission_intensity
            margin_percent = (margin / regulatory_limit * 100) if regulatory_limit > 0 else 0

            # Identify specific violations
            violations = []

            co2_limit = regulations.get('co2_kg_mwh', 180)
            if (co2_kg_hr / heat_output_mw) > co2_limit and heat_output_mw > 0:
                violations.append(f"CO2 emissions exceed limit: {co2_kg_hr/heat_output_mw:.2f} > {co2_limit} kg/MWh")

            nox_limit = regulations.get('nox_kg_mwh', 0.5)
            if (nox_kg_hr / heat_output_mw) > nox_limit and heat_output_mw > 0:
                violations.append(f"NOx emissions exceed limit: {nox_kg_hr/heat_output_mw:.2f} > {nox_limit} kg/MWh")

            timestamp = datetime.utcnow().isoformat()

            # Calculate provenance hash
            provenance_str = f"{emissions_data}{regulations}{compliance_status}{timestamp}"
            provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

            return ComplianceResult(
                total_emissions_kg_hr=round(total_emissions, 2),
                emission_intensity_kg_mwh=round(emission_intensity, 2),
                regulatory_limit_kg_mwh=regulatory_limit,
                compliance_status=compliance_status,
                margin_percent=round(margin_percent, 2),
                violations=violations,
                timestamp=timestamp,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            logger.error(f"Emissions compliance check failed: {str(e)}")
            raise

    @staticmethod
    def generate_kpi_dashboard(metrics: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """
        Generate KPI dashboard for process heat operations.

        Args:
            metrics: Performance metrics data

        Returns:
            KPI dashboard dictionary
        """
        try:
            timestamp = datetime.utcnow().isoformat()

            # Calculate KPIs
            dashboard = {
                'timestamp': timestamp,
                'operational_kpis': {
                    'overall_efficiency': metrics.get('thermal_efficiency', 0),
                    'capacity_utilization': metrics.get('capacity_utilization', 0),
                    'heat_recovery_rate': metrics.get('heat_recovery_rate', 0),
                    'availability': metrics.get('availability', 99.0)
                },
                'energy_kpis': {
                    'specific_energy_consumption': metrics.get('specific_energy', 0),
                    'fuel_mix': metrics.get('fuel_mix', {}),
                    'renewable_percentage': metrics.get('renewable_percentage', 0),
                    'energy_intensity': metrics.get('energy_intensity', 0)
                },
                'environmental_kpis': {
                    'co2_intensity': metrics.get('co2_intensity', 0),
                    'emissions_reduction_ytd': metrics.get('emissions_reduction_ytd', 0),
                    'compliance_score': metrics.get('compliance_score', 100),
                    'water_consumption': metrics.get('water_consumption', 0)
                },
                'financial_kpis': {
                    'energy_cost_per_mwh': metrics.get('energy_cost_per_mwh', 0),
                    'maintenance_cost_ratio': metrics.get('maintenance_cost_ratio', 0),
                    'cost_savings_mtd': metrics.get('cost_savings_mtd', 0),
                    'roi_percentage': metrics.get('roi_percentage', 0)
                },
                'trends': {
                    'efficiency_trend': ProcessHeatTools._calculate_trend(
                        metrics.get('efficiency_history', [])
                    ),
                    'emissions_trend': ProcessHeatTools._calculate_trend(
                        metrics.get('emissions_history', [])
                    ),
                    'cost_trend': ProcessHeatTools._calculate_trend(
                        metrics.get('cost_history', [])
                    )
                }
            }

            # Add provenance
            provenance_str = f"{metrics}{dashboard}{timestamp}"
            dashboard['provenance_hash'] = hashlib.sha256(provenance_str.encode()).hexdigest()

            return dashboard

        except Exception as e:
            logger.error(f"KPI dashboard generation failed: {str(e)}")
            raise

    @staticmethod
    def coordinate_process_heat_agents(
        agent_ids: List[str],
        commands: Dict[str, Any]
    ) -> Dict[str, Any]:  # type: ignore
        """
        Coordinate multiple process heat agents.

        Args:
            agent_ids: List of agent IDs to coordinate
            commands: Commands to distribute to agents

        Returns:
            Coordination result with task assignments
        """
        try:
            timestamp = datetime.utcnow().isoformat()

            # Create task assignments based on agent capabilities
            task_assignments = {}

            # Define agent capabilities (in production, this would come from registry)
            agent_capabilities = {
                'GL-002': ['boiler_control', 'steam_generation'],
                'GL-003': ['heat_recovery', 'waste_heat_utilization'],
                'GL-004': ['furnace_control', 'temperature_regulation'],
                'GL-005': ['emissions_monitoring', 'compliance_reporting']
            }

            # Assign tasks based on capabilities
            for task, parameters in commands.items():
                assigned = False
                for agent_id in agent_ids:
                    if agent_id in agent_capabilities:
                        capabilities = agent_capabilities[agent_id]
                        # Simple matching - in production would be more sophisticated
                        if any(cap in task.lower() for cap in capabilities):
                            if agent_id not in task_assignments:
                                task_assignments[agent_id] = []
                            task_assignments[agent_id].append({
                                'task': task,
                                'parameters': parameters,
                                'priority': parameters.get('priority', 'normal'),
                                'timeout_seconds': parameters.get('timeout', 60)
                            })
                            assigned = True
                            break

                # If not assigned to specific agent, assign to first available
                if not assigned and agent_ids:
                    agent_id = agent_ids[0]
                    if agent_id not in task_assignments:
                        task_assignments[agent_id] = []
                    task_assignments[agent_id].append({
                        'task': task,
                        'parameters': parameters,
                        'priority': 'normal',
                        'timeout_seconds': 60
                    })

            # Create coordination result
            coordination_result = {
                'timestamp': timestamp,
                'coordinated_agents': len(task_assignments),
                'total_tasks': sum(len(tasks) for tasks in task_assignments.values()),
                'task_assignments': task_assignments,
                'coordination_status': 'SUCCESS',
                'estimated_completion_time': (
                    datetime.utcnow() + timedelta(seconds=120)
                ).isoformat()
            }

            # Add provenance
            provenance_str = f"{agent_ids}{commands}{task_assignments}{timestamp}"
            coordination_result['provenance_hash'] = hashlib.sha256(provenance_str.encode()).hexdigest()

            return coordination_result

        except Exception as e:
            logger.error(f"Agent coordination failed: {str(e)}")
            raise

    @staticmethod
    def integrate_scada_data(scada_feed: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """
        Integrate and process SCADA data feed.

        Args:
            scada_feed: Raw SCADA data feed

        Returns:
            Processed SCADA data
        """
        try:
            timestamp = datetime.utcnow().isoformat()

            # Process SCADA tags
            processed_data = {
                'timestamp': timestamp,
                'data_points': {},
                'quality_metrics': {},
                'alarms': []
            }

            # Extract and validate data points
            for tag, value in scada_feed.get('tags', {}).items():
                # Validate data quality
                quality = scada_feed.get('quality', {}).get(tag, 100)

                if quality >= 90:  # Good quality threshold
                    processed_data['data_points'][tag] = {
                        'value': value,
                        'quality': quality,
                        'unit': scada_feed.get('units', {}).get(tag, 'unknown'),
                        'timestamp': scada_feed.get('timestamps', {}).get(tag, timestamp)
                    }
                else:
                    logger.warning(f"Low quality data for tag {tag}: {quality}%")

            # Check for alarms
            alarm_limits = scada_feed.get('alarm_limits', {})
            for tag, value in processed_data['data_points'].items():
                if tag in alarm_limits:
                    limits = alarm_limits[tag]
                    if value['value'] > limits.get('high', float('inf')):
                        processed_data['alarms'].append({
                            'tag': tag,
                            'type': 'HIGH',
                            'value': value['value'],
                            'limit': limits['high'],
                            'severity': 'WARNING'
                        })
                    elif value['value'] > limits.get('high_high', float('inf')):
                        processed_data['alarms'].append({
                            'tag': tag,
                            'type': 'HIGH_HIGH',
                            'value': value['value'],
                            'limit': limits['high_high'],
                            'severity': 'CRITICAL'
                        })

            # Calculate quality metrics
            total_tags = len(scada_feed.get('tags', {}))
            good_quality_tags = len(processed_data['data_points'])
            processed_data['quality_metrics'] = {
                'total_tags': total_tags,
                'good_quality_tags': good_quality_tags,
                'data_availability': (good_quality_tags / total_tags * 100) if total_tags > 0 else 0,
                'active_alarms': len(processed_data['alarms'])
            }

            # Add provenance
            provenance_str = f"{scada_feed}{processed_data}{timestamp}"
            processed_data['provenance_hash'] = hashlib.sha256(provenance_str.encode()).hexdigest()

            return processed_data

        except Exception as e:
            logger.error(f"SCADA data integration failed: {str(e)}")
            raise

    @staticmethod
    def integrate_erp_data(erp_feed: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """
        Integrate and process ERP data feed.

        Args:
            erp_feed: Raw ERP data feed

        Returns:
            Processed ERP data
        """
        try:
            timestamp = datetime.utcnow().isoformat()

            # Process ERP data
            processed_data = {
                'timestamp': timestamp,
                'cost_data': {},
                'material_data': {},
                'production_data': {},
                'maintenance_data': {}
            }

            # Extract cost data
            if 'costs' in erp_feed:
                for cost_center, costs in erp_feed['costs'].items():
                    processed_data['cost_data'][cost_center] = {
                        'fuel_cost': costs.get('fuel_cost', 0),
                        'electricity_cost': costs.get('electricity_cost', 0),
                        'maintenance_cost': costs.get('maintenance_cost', 0),
                        'total_cost': sum([
                            costs.get('fuel_cost', 0),
                            costs.get('electricity_cost', 0),
                            costs.get('maintenance_cost', 0)
                        ]),
                        'currency': costs.get('currency', 'USD'),
                        'period': costs.get('period', 'monthly')
                    }

            # Extract material data
            if 'materials' in erp_feed:
                for material_id, material in erp_feed['materials'].items():
                    processed_data['material_data'][material_id] = {
                        'description': material.get('description', ''),
                        'quantity': material.get('quantity', 0),
                        'unit': material.get('unit', 'kg'),
                        'unit_cost': material.get('unit_cost', 0),
                        'total_value': material.get('quantity', 0) * material.get('unit_cost', 0)
                    }

            # Extract production data
            if 'production' in erp_feed:
                processed_data['production_data'] = {
                    'planned_output': erp_feed['production'].get('planned_output', 0),
                    'actual_output': erp_feed['production'].get('actual_output', 0),
                    'efficiency': (
                        erp_feed['production'].get('actual_output', 0) /
                        erp_feed['production'].get('planned_output', 1) * 100
                    ) if erp_feed['production'].get('planned_output', 0) > 0 else 0,
                    'unit': erp_feed['production'].get('unit', 'tons')
                }

            # Extract maintenance data
            if 'maintenance' in erp_feed:
                processed_data['maintenance_data'] = {
                    'scheduled_tasks': erp_feed['maintenance'].get('scheduled_tasks', []),
                    'completed_tasks': erp_feed['maintenance'].get('completed_tasks', []),
                    'pending_tasks': erp_feed['maintenance'].get('pending_tasks', []),
                    'next_maintenance': erp_feed['maintenance'].get('next_maintenance', '')
                }

            # Calculate summary metrics
            processed_data['summary'] = {
                'total_costs': sum(
                    cd['total_cost'] for cd in processed_data['cost_data'].values()
                ),
                'material_value': sum(
                    md['total_value'] for md in processed_data['material_data'].values()
                ),
                'production_efficiency': processed_data['production_data'].get('efficiency', 0),
                'maintenance_completion_rate': (
                    len(processed_data['maintenance_data'].get('completed_tasks', [])) /
                    max(len(processed_data['maintenance_data'].get('scheduled_tasks', [])), 1) * 100
                )
            }

            # Add provenance
            provenance_str = f"{erp_feed}{processed_data}{timestamp}"
            processed_data['provenance_hash'] = hashlib.sha256(provenance_str.encode()).hexdigest()

            return processed_data

        except Exception as e:
            logger.error(f"ERP data integration failed: {str(e)}")
            raise

    @staticmethod
    def _calculate_trend(history: List[float]) -> str:
        """
        Calculate trend from historical data.

        Args:
            history: List of historical values

        Returns:
            Trend indicator (increasing, decreasing, stable)
        """
        if len(history) < 2:
            return "insufficient_data"

        # Simple linear regression
        x = np.arange(len(history))
        y = np.array(history)

        # Calculate slope
        n = len(history)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)

        # Determine trend
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"