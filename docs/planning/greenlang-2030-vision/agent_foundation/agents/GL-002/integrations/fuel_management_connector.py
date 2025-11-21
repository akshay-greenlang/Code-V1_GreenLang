# -*- coding: utf-8 -*-
"""
Fuel Management System Connector for GL-002 BoilerEfficiencyOptimizer

Integrates with fuel supply, metering, and quality monitoring systems.
Supports multi-fuel operations and automatic fuel switching optimization.

Features:
- Real-time fuel flow monitoring
- Fuel quality analysis integration
- Multi-fuel switching logic
- Cost tracking and optimization
- BTU/calorific value monitoring
- Tank level monitoring
- Supply chain integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
import statistics
from greenlang.determinism import DeterministicClock
from greenlang.determinism import FinancialDecimal

logger = logging.getLogger(__name__)


class FuelType(Enum):
    """Supported fuel types for boiler operation."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_2 = "fuel_oil_2"  # Light oil
    FUEL_OIL_6 = "fuel_oil_6"  # Heavy oil
    COAL = "coal"
    BIOMASS = "biomass"
    HYDROGEN = "hydrogen"
    PROPANE = "propane"
    BIOGAS = "biogas"
    WASTE_HEAT = "waste_heat"


class FuelQualityParameter(Enum):
    """Fuel quality parameters monitored."""
    HEATING_VALUE = "heating_value"  # BTU/unit or kJ/unit
    MOISTURE_CONTENT = "moisture_content"  # %
    SULFUR_CONTENT = "sulfur_content"  # %
    ASH_CONTENT = "ash_content"  # %
    DENSITY = "density"  # kg/m3 or lb/ft3
    VISCOSITY = "viscosity"  # cSt
    FLASH_POINT = "flash_point"  # °C
    CARBON_CONTENT = "carbon_content"  # %
    HYDROGEN_CONTENT = "hydrogen_content"  # %
    WOBBE_INDEX = "wobbe_index"  # For gas fuels


@dataclass
class FuelSpecification:
    """Fuel specification and properties."""
    fuel_type: FuelType
    heating_value_lower: float  # LHV in MJ/kg or MJ/m3
    heating_value_upper: float  # HHV in MJ/kg or MJ/m3
    density: float  # kg/m3 at reference conditions
    carbon_content: float  # % by mass
    hydrogen_content: float  # % by mass
    sulfur_content_max: float  # % by mass
    moisture_content_max: float  # % by mass
    cost_per_unit: float  # $/kg or $/m3
    co2_emission_factor: float  # kg CO2/kg fuel
    availability: bool = True
    supply_contract_limit: Optional[float] = None  # Max consumption rate


@dataclass
class FuelTank:
    """Fuel tank/storage configuration."""
    tank_id: str
    fuel_type: FuelType
    capacity: float  # m3 or kg
    current_level: float  # m3 or kg
    min_operating_level: float  # Minimum safe level
    max_fill_level: float  # Maximum safe level
    temperature: Optional[float] = None  # °C
    pressure: Optional[float] = None  # bar (for gas)
    location: str = ""
    last_refill_date: Optional[datetime] = None
    consumption_rate_avg: float = 0.0  # units/hour


@dataclass
class FuelFlowMeter:
    """Fuel flow meter configuration."""
    meter_id: str
    fuel_type: FuelType
    meter_type: str  # "coriolis", "turbine", "ultrasonic", "differential_pressure"
    min_flow: float  # kg/hr or m3/hr
    max_flow: float  # kg/hr or m3/hr
    accuracy_percent: float  # % of reading
    current_flow: float = 0.0
    total_consumption: float = 0.0  # Totalizer value
    temperature_compensated: bool = True
    pressure_compensated: bool = True


@dataclass
class FuelSupplyConfig:
    """Configuration for fuel supply system connection."""
    system_name: str
    connection_type: str  # "modbus", "opc_ua", "rest_api", "mqtt"
    host: str
    port: int
    polling_interval: int = 5  # seconds
    enable_quality_monitoring: bool = True
    enable_cost_tracking: bool = True
    enable_predictive_ordering: bool = False
    multi_fuel_enabled: bool = True
    auto_switching_enabled: bool = True


class FuelQualityAnalyzer:
    """
    Analyze fuel quality and calculate combustion characteristics.

    Provides real-time fuel quality scoring and efficiency predictions.
    """

    def __init__(self):
        """Initialize fuel quality analyzer."""
        self.quality_history = deque(maxlen=1000)
        self.quality_thresholds = {
            FuelType.NATURAL_GAS: {
                'heating_value_min': 35.0,  # MJ/m3
                'wobbe_index_range': (45, 55),
                'sulfur_max': 0.02
            },
            FuelType.FUEL_OIL_2: {
                'heating_value_min': 42.0,  # MJ/kg
                'sulfur_max': 0.5,
                'water_max': 0.1,
                'viscosity_range': (2.0, 6.0)
            },
            FuelType.COAL: {
                'heating_value_min': 20.0,  # MJ/kg
                'moisture_max': 15.0,
                'ash_max': 20.0,
                'sulfur_max': 3.0
            }
        }

    def analyze_quality(
        self,
        fuel_type: FuelType,
        quality_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze fuel quality and provide quality score.

        Args:
            fuel_type: Type of fuel
            quality_params: Quality parameters measured

        Returns:
            Quality analysis results with score (0-100)
        """
        score = 100.0
        issues = []
        recommendations = []

        thresholds = self.quality_thresholds.get(fuel_type, {})

        # Check heating value
        if 'heating_value' in quality_params:
            hv = quality_params['heating_value']
            min_hv = thresholds.get('heating_value_min', 0)
            if hv < min_hv:
                penalty = ((min_hv - hv) / min_hv) * 20
                score -= penalty
                issues.append(f"Low heating value: {hv:.1f} < {min_hv:.1f}")
                recommendations.append("Consider switching to higher quality fuel")

        # Check moisture content
        if 'moisture_content' in quality_params:
            moisture = quality_params['moisture_content']
            max_moisture = thresholds.get('moisture_max', 100)
            if moisture > max_moisture:
                penalty = ((moisture - max_moisture) / max_moisture) * 15
                score -= penalty
                issues.append(f"High moisture: {moisture:.1f}% > {max_moisture:.1f}%")
                recommendations.append("Increase fuel preheating or drying")

        # Check sulfur content
        if 'sulfur_content' in quality_params:
            sulfur = quality_params['sulfur_content']
            max_sulfur = thresholds.get('sulfur_max', 100)
            if sulfur > max_sulfur:
                penalty = ((sulfur - max_sulfur) / max_sulfur) * 10
                score -= penalty
                issues.append(f"High sulfur: {sulfur:.2f}% > {max_sulfur:.2f}%")
                recommendations.append("Adjust FGD system or switch fuel")

        # Calculate combustion efficiency impact
        efficiency_impact = self._calculate_efficiency_impact(fuel_type, quality_params)

        # Store in history
        analysis = {
            'timestamp': DeterministicClock.utcnow(),
            'fuel_type': fuel_type.value,
            'quality_score': max(0, score),
            'issues': issues,
            'recommendations': recommendations,
            'efficiency_impact': efficiency_impact,
            'parameters': quality_params
        }

        self.quality_history.append(analysis)

        return analysis

    def _calculate_efficiency_impact(
        self,
        fuel_type: FuelType,
        quality_params: Dict[str, float]
    ) -> float:
        """
        Calculate impact on boiler efficiency from fuel quality.

        Returns:
            Efficiency impact in percentage points
        """
        impact = 0.0

        # Moisture content reduces efficiency
        if 'moisture_content' in quality_params:
            moisture = quality_params['moisture_content']
            # Approximately 1% efficiency loss per 10% moisture
            impact -= moisture * 0.1

        # Low heating value reduces efficiency
        if 'heating_value' in quality_params:
            hv = quality_params['heating_value']
            thresholds = self.quality_thresholds.get(fuel_type, {})
            nominal_hv = thresholds.get('heating_value_min', hv)
            if hv < nominal_hv:
                impact -= ((nominal_hv - hv) / nominal_hv) * 5

        # High ash content reduces efficiency
        if 'ash_content' in quality_params:
            ash = quality_params['ash_content']
            # Approximately 0.5% efficiency loss per 5% ash
            impact -= ash * 0.1

        return round(impact, 2)


class FuelCostOptimizer:
    """
    Optimize fuel selection based on cost and efficiency.

    Considers fuel prices, availability, efficiency, and emissions.
    """

    def __init__(self):
        """Initialize fuel cost optimizer."""
        self.fuel_prices = {}
        self.switching_costs = {
            # Cost to switch between fuel types ($/switch)
            (FuelType.NATURAL_GAS, FuelType.FUEL_OIL_2): 500,
            (FuelType.FUEL_OIL_2, FuelType.NATURAL_GAS): 500,
            (FuelType.COAL, FuelType.BIOMASS): 1000,
        }
        self.emission_costs = {
            # $/ton CO2 equivalent
            'co2': 50,
            'so2': 500,
            'nox': 200
        }

    def calculate_fuel_cost(
        self,
        fuel_spec: FuelSpecification,
        consumption_rate: float,
        efficiency: float,
        duration_hours: float
    ) -> Dict[str, float]:
        """
        Calculate total fuel cost including emissions.

        Args:
            fuel_spec: Fuel specification
            consumption_rate: kg/hr or m3/hr
            efficiency: Boiler efficiency (0-1)
            duration_hours: Operating duration

        Returns:
            Cost breakdown
        """
        # Base fuel cost
        fuel_cost = consumption_rate * duration_hours * fuel_spec.cost_per_unit

        # Adjust for efficiency
        actual_consumption = consumption_rate / efficiency
        adjusted_fuel_cost = actual_consumption * duration_hours * fuel_spec.cost_per_unit

        # Calculate emission costs
        co2_emissions = actual_consumption * duration_hours * fuel_spec.co2_emission_factor
        co2_cost = co2_emissions * self.emission_costs.get('co2', 0) / 1000

        # Calculate SO2 emissions (based on sulfur content)
        so2_emissions = actual_consumption * duration_hours * fuel_spec.sulfur_content_max * 2  # S -> SO2
        so2_cost = so2_emissions * self.emission_costs.get('so2', 0) / 1000

        total_cost = adjusted_fuel_cost + co2_cost + so2_cost

        return {
            'fuel_cost': adjusted_fuel_cost,
            'co2_cost': co2_cost,
            'so2_cost': so2_cost,
            'total_cost': total_cost,
            'cost_per_mwh': total_cost / (consumption_rate * fuel_spec.heating_value_lower * efficiency * duration_hours / 3600)
        }

    def optimize_fuel_selection(
        self,
        available_fuels: List[FuelSpecification],
        load_profile: List[float],  # MW per hour
        current_fuel: FuelType
    ) -> Dict[str, Any]:
        """
        Optimize fuel selection for given load profile.

        Args:
            available_fuels: List of available fuel options
            load_profile: Hourly load requirements
            current_fuel: Currently used fuel

        Returns:
            Optimal fuel switching schedule
        """
        schedule = []
        total_cost = 0
        current = current_fuel

        for hour, load in enumerate(load_profile):
            best_fuel = None
            best_cost = FinancialDecimal.from_string('inf')

            for fuel_spec in available_fuels:
                if not fuel_spec.availability:
                    continue

                # Calculate consumption rate
                consumption_rate = load / (fuel_spec.heating_value_lower * 0.85)  # Assume 85% efficiency

                # Calculate hourly cost
                cost = self.calculate_fuel_cost(
                    fuel_spec,
                    consumption_rate,
                    0.85,
                    1.0
                )['total_cost']

                # Add switching cost if different fuel
                if fuel_spec.fuel_type != current:
                    switch_cost = self.switching_costs.get(
                        (current, fuel_spec.fuel_type), 1000
                    )
                    cost += switch_cost / 24  # Amortize over day

                if cost < best_cost:
                    best_cost = cost
                    best_fuel = fuel_spec

            if best_fuel:
                schedule.append({
                    'hour': hour,
                    'fuel': best_fuel.fuel_type.value,
                    'load': load,
                    'cost': best_cost
                })
                total_cost += best_cost
                current = best_fuel.fuel_type

        return {
            'schedule': schedule,
            'total_cost': total_cost,
            'average_cost_per_mwh': total_cost / sum(load_profile) if load_profile else 0,
            'fuel_switches': len(set(s['fuel'] for s in schedule)) - 1
        }


class FuelManagementConnector:
    """
    Main fuel management system connector.

    Integrates fuel supply, quality, and optimization systems.
    """

    def __init__(self, config: FuelSupplyConfig):
        """Initialize fuel management connector."""
        self.config = config
        self.connected = False
        self.fuel_specs: Dict[FuelType, FuelSpecification] = {}
        self.fuel_tanks: Dict[str, FuelTank] = {}
        self.flow_meters: Dict[str, FuelFlowMeter] = {}
        self.quality_analyzer = FuelQualityAnalyzer()
        self.cost_optimizer = FuelCostOptimizer()
        self.data_buffer = deque(maxlen=10000)
        self._setup_default_fuels()

    def _setup_default_fuels(self):
        """Setup default fuel specifications."""
        self.fuel_specs = {
            FuelType.NATURAL_GAS: FuelSpecification(
                fuel_type=FuelType.NATURAL_GAS,
                heating_value_lower=48.0,  # MJ/m3
                heating_value_upper=53.0,
                density=0.75,  # kg/m3
                carbon_content=75.0,
                hydrogen_content=25.0,
                sulfur_content_max=0.01,
                moisture_content_max=0.0,
                cost_per_unit=0.35,  # $/m3
                co2_emission_factor=1.95  # kg CO2/m3
            ),
            FuelType.FUEL_OIL_2: FuelSpecification(
                fuel_type=FuelType.FUEL_OIL_2,
                heating_value_lower=42.5,  # MJ/kg
                heating_value_upper=45.5,
                density=850.0,  # kg/m3
                carbon_content=86.0,
                hydrogen_content=13.0,
                sulfur_content_max=0.5,
                moisture_content_max=0.1,
                cost_per_unit=0.75,  # $/kg
                co2_emission_factor=3.15  # kg CO2/kg
            ),
            FuelType.BIOMASS: FuelSpecification(
                fuel_type=FuelType.BIOMASS,
                heating_value_lower=15.0,  # MJ/kg
                heating_value_upper=19.0,
                density=250.0,  # kg/m3
                carbon_content=48.0,
                hydrogen_content=6.0,
                sulfur_content_max=0.1,
                moisture_content_max=20.0,
                cost_per_unit=0.12,  # $/kg
                co2_emission_factor=0.0  # Carbon neutral
            )
        }

    async def connect(self) -> bool:
        """Connect to fuel management system."""
        try:
            # Simulate connection based on type
            if self.config.connection_type == "modbus":
                logger.info(f"Connecting to fuel system via Modbus at {self.config.host}:{self.config.port}")
            elif self.config.connection_type == "opc_ua":
                logger.info(f"Connecting to fuel system via OPC UA at {self.config.host}:{self.config.port}")
            elif self.config.connection_type == "rest_api":
                logger.info(f"Connecting to fuel system API at {self.config.host}:{self.config.port}")

            self.connected = True

            # Initialize fuel tanks
            await self._initialize_fuel_tanks()

            # Initialize flow meters
            await self._initialize_flow_meters()

            logger.info(f"Connected to fuel management system: {self.config.system_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to fuel system: {e}")
            return False

    async def _initialize_fuel_tanks(self):
        """Initialize fuel tank monitoring."""
        self.fuel_tanks = {
            'gas_supply_1': FuelTank(
                tank_id='gas_supply_1',
                fuel_type=FuelType.NATURAL_GAS,
                capacity=10000,  # m3
                current_level=7500,
                min_operating_level=1000,
                max_fill_level=9500,
                pressure=50,  # bar
                location='Main gas station'
            ),
            'oil_tank_1': FuelTank(
                tank_id='oil_tank_1',
                fuel_type=FuelType.FUEL_OIL_2,
                capacity=100000,  # liters
                current_level=65000,
                min_operating_level=10000,
                max_fill_level=95000,
                temperature=25,
                location='Tank farm A'
            ),
            'biomass_silo_1': FuelTank(
                tank_id='biomass_silo_1',
                fuel_type=FuelType.BIOMASS,
                capacity=500000,  # kg
                current_level=250000,
                min_operating_level=50000,
                max_fill_level=450000,
                location='Biomass storage'
            )
        }

    async def _initialize_flow_meters(self):
        """Initialize fuel flow meters."""
        self.flow_meters = {
            'gas_meter_1': FuelFlowMeter(
                meter_id='gas_meter_1',
                fuel_type=FuelType.NATURAL_GAS,
                meter_type='ultrasonic',
                min_flow=10,  # m3/hr
                max_flow=5000,
                accuracy_percent=0.5
            ),
            'oil_meter_1': FuelFlowMeter(
                meter_id='oil_meter_1',
                fuel_type=FuelType.FUEL_OIL_2,
                meter_type='coriolis',
                min_flow=10,  # kg/hr
                max_flow=10000,
                accuracy_percent=0.2
            ),
            'biomass_meter_1': FuelFlowMeter(
                meter_id='biomass_meter_1',
                fuel_type=FuelType.BIOMASS,
                meter_type='weighbelt',
                min_flow=100,  # kg/hr
                max_flow=20000,
                accuracy_percent=1.0
            )
        }

    async def read_fuel_flow(self, meter_id: str) -> Optional[Dict[str, Any]]:
        """
        Read current fuel flow from meter.

        Args:
            meter_id: Flow meter identifier

        Returns:
            Flow reading with metadata
        """
        if not self.connected or meter_id not in self.flow_meters:
            return None

        meter = self.flow_meters[meter_id]

        # Simulate flow reading
        import random
        flow = random.uniform(meter.min_flow, meter.max_flow * 0.7)
        meter.current_flow = flow
        meter.total_consumption += flow * (self.config.polling_interval / 3600)

        reading = {
            'meter_id': meter_id,
            'fuel_type': meter.fuel_type.value,
            'flow_rate': flow,
            'unit': 'm3/hr' if meter.fuel_type == FuelType.NATURAL_GAS else 'kg/hr',
            'total_consumption': meter.total_consumption,
            'timestamp': DeterministicClock.utcnow().isoformat()
        }

        # Store in buffer
        self.data_buffer.append(reading)

        return reading

    async def read_fuel_quality(self, fuel_type: FuelType) -> Optional[Dict[str, Any]]:
        """
        Read current fuel quality parameters.

        Args:
            fuel_type: Type of fuel to check

        Returns:
            Quality parameters and analysis
        """
        if not self.connected or not self.config.enable_quality_monitoring:
            return None

        # Simulate quality reading
        import random

        quality_params = {}

        if fuel_type == FuelType.NATURAL_GAS:
            quality_params = {
                'heating_value': random.uniform(45, 52),  # MJ/m3
                'wobbe_index': random.uniform(48, 53),
                'methane_content': random.uniform(85, 95),  # %
                'sulfur_content': random.uniform(0.001, 0.01)  # %
            }
        elif fuel_type == FuelType.FUEL_OIL_2:
            quality_params = {
                'heating_value': random.uniform(41, 44),  # MJ/kg
                'density': random.uniform(840, 860),  # kg/m3
                'viscosity': random.uniform(3, 5),  # cSt
                'sulfur_content': random.uniform(0.1, 0.4),  # %
                'water_content': random.uniform(0.01, 0.08)  # %
            }
        elif fuel_type == FuelType.BIOMASS:
            quality_params = {
                'heating_value': random.uniform(14, 18),  # MJ/kg
                'moisture_content': random.uniform(10, 25),  # %
                'ash_content': random.uniform(1, 5),  # %
                'volatile_matter': random.uniform(70, 80)  # %
            }

        # Analyze quality
        analysis = self.quality_analyzer.analyze_quality(fuel_type, quality_params)

        return {
            'fuel_type': fuel_type.value,
            'parameters': quality_params,
            'analysis': analysis,
            'timestamp': DeterministicClock.utcnow().isoformat()
        }

    async def get_tank_levels(self) -> Dict[str, Any]:
        """Get current fuel tank levels and supply status."""
        levels = {}

        for tank_id, tank in self.fuel_tanks.items():
            # Calculate days of supply remaining
            if tank.consumption_rate_avg > 0:
                days_remaining = (tank.current_level - tank.min_operating_level) / (tank.consumption_rate_avg * 24)
            else:
                days_remaining = float('inf')

            levels[tank_id] = {
                'fuel_type': tank.fuel_type.value,
                'current_level': tank.current_level,
                'capacity': tank.capacity,
                'percentage': (tank.current_level / tank.capacity) * 100,
                'days_remaining': days_remaining,
                'requires_refill': tank.current_level < (tank.min_operating_level * 1.5),
                'location': tank.location
            }

        return levels

    async def calculate_fuel_cost(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Calculate fuel costs for a time period.

        Args:
            start_time: Period start
            end_time: Period end

        Returns:
            Cost breakdown by fuel type
        """
        if not self.config.enable_cost_tracking:
            return {}

        costs = {}
        total_cost = 0
        total_energy = 0

        for meter_id, meter in self.flow_meters.items():
            fuel_type = meter.fuel_type
            fuel_spec = self.fuel_specs.get(fuel_type)

            if not fuel_spec:
                continue

            # Calculate consumption (simulated)
            hours = (end_time - start_time).total_seconds() / 3600
            avg_flow = (meter.min_flow + meter.max_flow) / 3  # Assumed average
            consumption = avg_flow * hours

            # Calculate cost
            fuel_cost = consumption * fuel_spec.cost_per_unit
            energy_delivered = consumption * fuel_spec.heating_value_lower

            costs[fuel_type.value] = {
                'consumption': consumption,
                'unit': 'm3' if fuel_type == FuelType.NATURAL_GAS else 'kg',
                'cost': fuel_cost,
                'energy_delivered': energy_delivered,  # MJ
                'cost_per_mj': fuel_cost / energy_delivered if energy_delivered > 0 else 0
            }

            total_cost += fuel_cost
            total_energy += energy_delivered

        return {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'fuel_costs': costs,
            'total_cost': total_cost,
            'total_energy': total_energy,
            'average_cost_per_mj': total_cost / total_energy if total_energy > 0 else 0
        }

    async def optimize_fuel_mix(
        self,
        load_forecast: List[float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize fuel mix for forecasted load.

        Args:
            load_forecast: Hourly load forecast (MW)
            constraints: Optional constraints (emissions, costs, etc.)

        Returns:
            Optimized fuel schedule
        """
        if not self.config.multi_fuel_enabled:
            return {'error': 'Multi-fuel operation not enabled'}

        # Get available fuels
        available_fuels = [
            spec for spec in self.fuel_specs.values()
            if spec.availability
        ]

        # Get current fuel type (from largest flow)
        current_fuel = FuelType.NATURAL_GAS  # Default

        for meter in self.flow_meters.values():
            if meter.current_flow > 0:
                current_fuel = meter.fuel_type
                break

        # Optimize fuel selection
        optimization = self.cost_optimizer.optimize_fuel_selection(
            available_fuels,
            load_forecast,
            current_fuel
        )

        # Apply constraints if provided
        if constraints:
            if 'max_emissions' in constraints:
                # Filter schedule to meet emission constraints
                pass
            if 'max_cost' in constraints:
                # Filter schedule to meet cost constraints
                pass

        return optimization

    async def execute_fuel_switch(
        self,
        from_fuel: FuelType,
        to_fuel: FuelType,
        ramp_time_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Execute fuel switching sequence.

        Args:
            from_fuel: Current fuel type
            to_fuel: Target fuel type
            ramp_time_minutes: Time for transition

        Returns:
            Switch execution status
        """
        if not self.config.auto_switching_enabled:
            return {
                'success': False,
                'error': 'Automatic fuel switching not enabled'
            }

        # Validate fuel availability
        if to_fuel not in self.fuel_specs or not self.fuel_specs[to_fuel].availability:
            return {
                'success': False,
                'error': f'Fuel {to_fuel.value} not available'
            }

        # Check tank levels
        tank_available = False
        for tank in self.fuel_tanks.values():
            if tank.fuel_type == to_fuel and tank.current_level > tank.min_operating_level:
                tank_available = True
                break

        if not tank_available:
            return {
                'success': False,
                'error': f'Insufficient {to_fuel.value} in tanks'
            }

        # Simulate switching sequence
        sequence = [
            {'time': 0, 'action': f'Initiate switch from {from_fuel.value} to {to_fuel.value}'},
            {'time': 5, 'action': 'Verify fuel supply availability'},
            {'time': 10, 'action': 'Start fuel preheating/preparation'},
            {'time': 15, 'action': 'Begin fuel flow transition'},
            {'time': ramp_time_minutes - 5, 'action': 'Stabilize combustion'},
            {'time': ramp_time_minutes, 'action': 'Switch complete'}
        ]

        logger.info(f"Executing fuel switch: {from_fuel.value} -> {to_fuel.value}")

        return {
            'success': True,
            'from_fuel': from_fuel.value,
            'to_fuel': to_fuel.value,
            'ramp_time': ramp_time_minutes,
            'sequence': sequence,
            'estimated_cost': self.cost_optimizer.switching_costs.get(
                (from_fuel, to_fuel), 1000
            )
        }

    async def disconnect(self):
        """Disconnect from fuel management system."""
        if self.connected:
            self.connected = False
            logger.info(f"Disconnected from fuel management system: {self.config.system_name}")


# Example usage
async def main():
    """Example usage of fuel management connector."""

    # Configure fuel management connection
    config = FuelSupplyConfig(
        system_name="Boiler_Fuel_System_1",
        connection_type="modbus",
        host="192.168.1.110",
        port=502,
        enable_quality_monitoring=True,
        enable_cost_tracking=True,
        multi_fuel_enabled=True,
        auto_switching_enabled=True
    )

    # Initialize connector
    connector = FuelManagementConnector(config)

    # Connect to system
    await connector.connect()

    # Read fuel flows
    for meter_id in connector.flow_meters.keys():
        flow = await connector.read_fuel_flow(meter_id)
        print(f"Flow reading: {json.dumps(flow, indent=2)}")

    # Check fuel quality
    for fuel_type in [FuelType.NATURAL_GAS, FuelType.FUEL_OIL_2]:
        quality = await connector.read_fuel_quality(fuel_type)
        if quality:
            print(f"Fuel quality for {fuel_type.value}:")
            print(f"  Quality score: {quality['analysis']['quality_score']:.1f}/100")
            print(f"  Efficiency impact: {quality['analysis']['efficiency_impact']:.1f}%")

    # Check tank levels
    levels = await connector.get_tank_levels()
    print(f"\nTank levels: {json.dumps(levels, indent=2)}")

    # Calculate fuel costs
    end_time = DeterministicClock.utcnow()
    start_time = end_time - timedelta(hours=24)
    costs = await connector.calculate_fuel_cost(start_time, end_time)
    print(f"\n24-hour fuel costs: ${costs.get('total_cost', 0):.2f}")

    # Optimize fuel mix for next 24 hours
    load_forecast = [100, 95, 90, 85, 80, 85, 95, 110, 120, 125,
                     130, 135, 140, 135, 130, 125, 120, 115, 110, 105,
                     100, 95, 90, 85]  # MW per hour

    optimization = await connector.optimize_fuel_mix(load_forecast)
    print(f"\nOptimized fuel schedule:")
    print(f"  Total cost: ${optimization['total_cost']:.2f}")
    print(f"  Fuel switches: {optimization['fuel_switches']}")

    # Test fuel switching
    switch_result = await connector.execute_fuel_switch(
        FuelType.NATURAL_GAS,
        FuelType.FUEL_OIL_2,
        ramp_time_minutes=30
    )
    print(f"\nFuel switch result: {json.dumps(switch_result, indent=2)}")

    # Disconnect
    await connector.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())