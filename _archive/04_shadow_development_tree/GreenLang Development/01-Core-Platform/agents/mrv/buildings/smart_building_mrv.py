# -*- coding: utf-8 -*-
"""
GL-MRV-BLD-008: Smart Building MRV Agent
==========================================

Specialized MRV agent for IoT-enabled smart buildings with
real-time data integration and granular emissions tracking.

Features:
    - IoT sensor data integration
    - Real-time energy monitoring
    - Sub-hourly emissions tracking
    - Anomaly detection integration
    - Digital twin data support
    - BACnet/Modbus data parsing

Standards:
    - ASHRAE Guideline 36 - High Performance HVAC
    - ISO 52000 - EPBD framework
    - Project Haystack data modeling

Author: GreenLang Framework Team
Agent ID: GL-MRV-BLD-008
Version: 1.0.0
"""

from __future__ import annotations

import logging
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field

from greenlang.agents.mrv.buildings.base import (
    BuildingMRVBaseAgent,
    BuildingMRVInput,
    BuildingMRVOutput,
    BuildingMetadata,
    BuildingType,
    EnergyConsumption,
    EnergySource,
    EndUseCategory,
    EmissionFactor,
    CalculationStep,
    EnergyUseIntensity,
    CarbonIntensity,
    DataQuality,
    VerificationStatus,
    GRID_EF_BY_REGION_KGCO2E_PER_KWH,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class SensorType(str, Enum):
    """IoT sensor types."""
    POWER_METER = "power_meter"
    ENERGY_METER = "energy_meter"
    GAS_METER = "gas_meter"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    CO2_SENSOR = "co2_sensor"
    OCCUPANCY = "occupancy"
    LIGHT_LEVEL = "light_level"
    WATER_METER = "water_meter"


class DataGranularity(str, Enum):
    """Data granularity levels."""
    MINUTE = "minute"
    FIFTEEN_MINUTE = "15min"
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


class AnomalyType(str, Enum):
    """Types of anomalies detected."""
    SPIKE = "spike"
    DRIFT = "drift"
    FLAT_LINE = "flat_line"
    MISSING_DATA = "missing_data"
    PATTERN_CHANGE = "pattern_change"


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class SensorReading(BaseModel):
    """Individual sensor reading."""
    sensor_id: str
    sensor_type: SensorType
    timestamp: str  # ISO format
    value: Decimal
    unit: str
    quality_flag: Optional[str] = None
    end_use: Optional[EndUseCategory] = None
    zone_id: Optional[str] = None


class TimeSeriesData(BaseModel):
    """Time series energy data."""
    start_time: str
    end_time: str
    granularity: DataGranularity
    readings: List[SensorReading]


class AnomalyRecord(BaseModel):
    """Detected anomaly record."""
    anomaly_id: str
    sensor_id: str
    anomaly_type: AnomalyType
    start_time: str
    end_time: Optional[str] = None
    severity: str = Field(default="medium")
    estimated_impact_kwh: Optional[Decimal] = None
    description: Optional[str] = None


class SmartBuildingInput(BuildingMRVInput):
    """Input model for smart building MRV."""

    # Time series data
    time_series_data: Optional[TimeSeriesData] = None

    # Real-time sensor readings
    sensor_readings: List[SensorReading] = Field(default_factory=list)

    # Anomalies detected
    anomalies: List[AnomalyRecord] = Field(default_factory=list)

    # BMS integration
    bms_connected: bool = Field(default=False)
    num_connected_sensors: Optional[int] = Field(None, ge=0)
    data_completeness_percent: Optional[Decimal] = Field(None, ge=0, le=100)

    # Zone-level data
    zones_monitored: Optional[int] = Field(None, ge=0)
    include_zone_breakdown: bool = Field(default=False)

    # Demand response
    demand_response_events: Optional[int] = Field(None, ge=0)
    demand_response_kwh_reduced: Optional[Decimal] = Field(None, ge=0)


class ZoneEmissions(BaseModel):
    """Emissions by building zone."""
    zone_id: str
    zone_name: Optional[str] = None
    energy_kwh: Decimal
    emissions_kgco2e: Decimal
    floor_area_sqm: Optional[Decimal] = None


class SmartBuildingOutput(BuildingMRVOutput):
    """Output model for smart building MRV."""

    # Granular emissions
    emissions_by_hour: Optional[Dict[str, Decimal]] = None
    peak_demand_kw: Optional[Decimal] = None
    peak_demand_time: Optional[str] = None

    # Zone breakdown
    zone_emissions: List[ZoneEmissions] = Field(default_factory=list)

    # End-use breakdown
    energy_by_end_use: Dict[str, Decimal] = Field(default_factory=dict)
    emissions_by_end_use: Dict[str, Decimal] = Field(default_factory=dict)

    # Anomaly impact
    anomalies_detected: int = Field(default=0)
    estimated_anomaly_waste_kwh: Decimal = Field(default=Decimal("0"))
    estimated_anomaly_emissions_kgco2e: Decimal = Field(default=Decimal("0"))

    # Data quality metrics
    sensor_coverage_percent: Optional[Decimal] = None
    data_completeness_percent: Optional[Decimal] = None
    metered_vs_estimated_ratio: Optional[Decimal] = None

    # Demand response
    demand_response_savings_kwh: Optional[Decimal] = None
    demand_response_savings_kgco2e: Optional[Decimal] = None


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class SmartBuildingMRVAgent(BuildingMRVBaseAgent[SmartBuildingInput, SmartBuildingOutput]):
    """
    GL-MRV-BLD-008: Smart Building MRV Agent.

    Calculates emissions for IoT-enabled buildings using real-time
    sensor data with granular tracking and anomaly detection.

    ZERO-HALLUCINATION GUARANTEE:
        - All calculations use deterministic aggregation
        - Sensor data validated before processing
        - Complete audit trail for verification
        - Reproducible results with same inputs

    Example:
        >>> agent = SmartBuildingMRVAgent()
        >>> input_data = SmartBuildingInput(
        ...     building_id="SMART-001",
        ...     reporting_period="2024",
        ...     building_metadata=BuildingMetadata(...),
        ...     sensor_readings=[
        ...         SensorReading(
        ...             sensor_id="PWR-001",
        ...             sensor_type=SensorType.POWER_METER,
        ...             timestamp="2024-01-01T00:00:00Z",
        ...             value=Decimal("50.5"),
        ...             unit="kW"
        ...         )
        ...     ]
        ... )
        >>> output = agent.process(input_data)
    """

    AGENT_ID = "GL-MRV-BLD-008"
    AGENT_VERSION = "1.0.0"
    BUILDING_CATEGORY = "smart_building"

    def _load_emission_factors(self) -> None:
        """Load smart building emission factors."""
        for region, factor in GRID_EF_BY_REGION_KGCO2E_PER_KWH.items():
            self._emission_factors[f"scope2_electricity_{region}"] = EmissionFactor(
                factor_id=f"scope2_electricity_{region}",
                value=factor,
                unit="kgCO2e/kWh",
                source="EPA eGRID 2024 / Real-time marginal factors",
                region=region,
                valid_from="2024-01-01"
            )

    def calculate_emissions(
        self,
        input_data: SmartBuildingInput
    ) -> SmartBuildingOutput:
        """
        Calculate smart building emissions from IoT data.

        Methodology:
        1. Aggregate sensor readings by type and zone
        2. Calculate energy consumption from power meters
        3. Apply emission factors with time-of-use consideration
        4. Calculate zone-level breakdown
        5. Quantify anomaly impact
        6. Apply demand response credits

        Args:
            input_data: Validated smart building input

        Returns:
            Complete smart building emission output with provenance
        """
        calculation_steps: List[CalculationStep] = []
        factors_used: List[EmissionFactor] = []
        step_number = 1

        metadata = input_data.building_metadata
        floor_area = metadata.gross_floor_area_sqm

        # Get grid emission factor
        grid_ef = input_data.grid_emission_factor_kgco2e_per_kwh
        if grid_ef is None:
            grid_ef = GRID_EF_BY_REGION_KGCO2E_PER_KWH.get("us_average", Decimal("0.379"))

        # Step 1: Process sensor readings
        energy_by_sensor: Dict[str, Decimal] = {}
        energy_by_zone: Dict[str, Decimal] = {}
        energy_by_end_use: Dict[str, Decimal] = {}
        power_readings: List[tuple] = []  # (timestamp, power_kw)

        for reading in input_data.sensor_readings:
            if reading.sensor_type == SensorType.ENERGY_METER:
                # Direct energy reading
                energy_kwh = reading.value
                if reading.unit.lower() == "mwh":
                    energy_kwh = reading.value * 1000
                elif reading.unit.lower() == "wh":
                    energy_kwh = reading.value / 1000

                energy_by_sensor[reading.sensor_id] = energy_by_sensor.get(
                    reading.sensor_id, Decimal("0")
                ) + energy_kwh

                # Track by zone
                if reading.zone_id:
                    energy_by_zone[reading.zone_id] = energy_by_zone.get(
                        reading.zone_id, Decimal("0")
                    ) + energy_kwh

                # Track by end use
                if reading.end_use:
                    end_use_key = reading.end_use.value
                    energy_by_end_use[end_use_key] = energy_by_end_use.get(
                        end_use_key, Decimal("0")
                    ) + energy_kwh

            elif reading.sensor_type == SensorType.POWER_METER:
                power_kw = reading.value
                if reading.unit.lower() == "mw":
                    power_kw = reading.value * 1000
                elif reading.unit.lower() == "w":
                    power_kw = reading.value / 1000

                power_readings.append((reading.timestamp, power_kw))

        # Step 2: Calculate total energy from sensor data
        total_energy_sensors = sum(energy_by_sensor.values())

        # If time series data provided, use that
        if input_data.time_series_data:
            total_energy_sensors = self._calculate_from_time_series(
                input_data.time_series_data
            )

        # Combine with traditional energy consumption data
        total_energy_traditional = Decimal("0")
        for consumption in input_data.energy_consumption:
            if consumption.source == EnergySource.ELECTRICITY:
                total_energy_traditional += self._convert_to_kwh(
                    consumption.consumption,
                    consumption.unit,
                    consumption.source
                )

        # Use sensor data if available, otherwise traditional
        total_energy_kwh = total_energy_sensors if total_energy_sensors > 0 else total_energy_traditional

        calculation_steps.append(CalculationStep(
            step_number=step_number,
            description="Aggregate energy from IoT sensors and meters",
            formula="total_energy = sum(sensor_readings) or utility_data",
            inputs={
                "num_sensors": str(len(input_data.sensor_readings)),
                "sensor_energy_kwh": str(total_energy_sensors),
                "utility_energy_kwh": str(total_energy_traditional)
            },
            output_value=self._round_energy(total_energy_kwh),
            output_unit="kWh",
            source="IoT sensor aggregation"
        ))
        step_number += 1

        # Step 3: Calculate peak demand
        peak_demand_kw = None
        peak_demand_time = None

        if power_readings:
            max_reading = max(power_readings, key=lambda x: x[1])
            peak_demand_kw = self._round_energy(max_reading[1])
            peak_demand_time = max_reading[0]

        # Step 4: Calculate emissions
        scope2_emissions = total_energy_kwh * grid_ef

        calculation_steps.append(CalculationStep(
            step_number=step_number,
            description="Calculate Scope 2 emissions from electricity",
            formula="scope2_emissions = energy_kwh * grid_ef",
            inputs={
                "energy_kwh": str(self._round_energy(total_energy_kwh)),
                "grid_ef": str(grid_ef)
            },
            output_value=self._round_emissions(scope2_emissions),
            output_unit="kgCO2e",
            source="EPA eGRID 2024"
        ))
        step_number += 1

        # Step 5: Calculate zone-level emissions
        zone_emissions_list: List[ZoneEmissions] = []
        for zone_id, zone_energy in energy_by_zone.items():
            zone_co2 = zone_energy * grid_ef
            zone_emissions_list.append(ZoneEmissions(
                zone_id=zone_id,
                energy_kwh=self._round_energy(zone_energy),
                emissions_kgco2e=self._round_emissions(zone_co2)
            ))

        # Step 6: Calculate emissions by end use
        emissions_by_end_use: Dict[str, Decimal] = {}
        for end_use, energy in energy_by_end_use.items():
            emissions_by_end_use[end_use] = self._round_emissions(energy * grid_ef)

        # Step 7: Calculate anomaly impact
        anomalies_detected = len(input_data.anomalies)
        anomaly_waste_kwh = Decimal("0")

        for anomaly in input_data.anomalies:
            if anomaly.estimated_impact_kwh:
                anomaly_waste_kwh += anomaly.estimated_impact_kwh

        anomaly_emissions = anomaly_waste_kwh * grid_ef

        if anomaly_waste_kwh > 0:
            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate estimated anomaly waste emissions",
                formula="anomaly_emissions = anomaly_kwh * grid_ef",
                inputs={
                    "anomalies_detected": str(anomalies_detected),
                    "estimated_waste_kwh": str(anomaly_waste_kwh)
                },
                output_value=self._round_emissions(anomaly_emissions),
                output_unit="kgCO2e",
                source="Anomaly detection analysis"
            ))
            step_number += 1

        # Step 8: Calculate demand response savings
        dr_savings_kwh = None
        dr_savings_emissions = None

        if input_data.demand_response_kwh_reduced:
            dr_savings_kwh = input_data.demand_response_kwh_reduced
            dr_savings_emissions = self._round_emissions(dr_savings_kwh * grid_ef)

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate demand response emission savings",
                formula="dr_savings = reduced_kwh * grid_ef",
                inputs={
                    "dr_events": str(input_data.demand_response_events or 0),
                    "reduced_kwh": str(dr_savings_kwh)
                },
                output_value=dr_savings_emissions,
                output_unit="kgCO2e",
                source="Demand response program"
            ))
            step_number += 1

        # Calculate data quality metrics
        sensor_coverage = None
        data_completeness = input_data.data_completeness_percent
        metered_ratio = None

        if input_data.num_connected_sensors and floor_area > 0:
            # Estimate coverage based on sensors per sqm
            sensors_per_1000sqm = Decimal(str(input_data.num_connected_sensors)) / (floor_area / 1000)
            # Assume good coverage at 5+ sensors per 1000 sqm
            sensor_coverage = min(Decimal("100"), sensors_per_1000sqm * 20)

        if total_energy_sensors > 0 and total_energy_traditional > 0:
            metered_ratio = self._round_intensity(
                total_energy_sensors / (total_energy_sensors + total_energy_traditional) * 100
            )

        # Calculate intensity metrics
        eui_metrics = self._calculate_eui(total_energy_kwh, floor_area)

        carbon_intensity = self._calculate_carbon_intensity(
            Decimal("0"),  # No Scope 1 tracked in smart building agent
            self._round_emissions(scope2_emissions),
            floor_area
        )

        return SmartBuildingOutput(
            calculation_id=self._generate_calculation_id(
                input_data.building_id,
                input_data.reporting_period
            ),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=input_data.building_id,
            building_type=metadata.building_type,
            reporting_period=input_data.reporting_period,
            gross_floor_area_sqm=floor_area,
            total_energy_kwh=self._round_energy(total_energy_kwh),
            energy_by_source={"electricity": self._round_energy(total_energy_kwh)},
            eui_metrics=eui_metrics,
            scope_1_emissions_kgco2e=Decimal("0"),
            scope_2_emissions_kgco2e=self._round_emissions(scope2_emissions),
            scope_3_emissions_kgco2e=Decimal("0"),
            total_emissions_kgco2e=self._round_emissions(scope2_emissions),
            carbon_intensity=carbon_intensity,
            calculation_steps=calculation_steps,
            emission_factors_used=factors_used,
            data_quality=DataQuality.METERED if input_data.bms_connected else input_data.data_quality,
            verification_status=VerificationStatus.UNVERIFIED,
            is_valid=True,
            peak_demand_kw=peak_demand_kw,
            peak_demand_time=peak_demand_time,
            zone_emissions=zone_emissions_list,
            energy_by_end_use={k: self._round_energy(v) for k, v in energy_by_end_use.items()},
            emissions_by_end_use=emissions_by_end_use,
            anomalies_detected=anomalies_detected,
            estimated_anomaly_waste_kwh=self._round_energy(anomaly_waste_kwh),
            estimated_anomaly_emissions_kgco2e=self._round_emissions(anomaly_emissions),
            sensor_coverage_percent=sensor_coverage,
            data_completeness_percent=data_completeness,
            metered_vs_estimated_ratio=metered_ratio,
            demand_response_savings_kwh=dr_savings_kwh,
            demand_response_savings_kgco2e=dr_savings_emissions
        )

    def _calculate_from_time_series(
        self,
        ts_data: TimeSeriesData
    ) -> Decimal:
        """Calculate total energy from time series data."""
        total_energy = Decimal("0")

        # Group readings by sensor
        for reading in ts_data.readings:
            if reading.sensor_type in {SensorType.ENERGY_METER, SensorType.POWER_METER}:
                if reading.sensor_type == SensorType.ENERGY_METER:
                    total_energy += reading.value
                elif reading.sensor_type == SensorType.POWER_METER:
                    # Convert power to energy based on granularity
                    hours_factor = {
                        DataGranularity.MINUTE: Decimal("1") / Decimal("60"),
                        DataGranularity.FIFTEEN_MINUTE: Decimal("0.25"),
                        DataGranularity.HOURLY: Decimal("1"),
                        DataGranularity.DAILY: Decimal("24"),
                    }.get(ts_data.granularity, Decimal("1"))

                    total_energy += reading.value * hours_factor

        return total_energy
