"""
Fugitive Emissions Module - GL-010 EmissionsGuardian

This module implements EPA Method 21 leak detection and repair (LDAR) program
management for fugitive emissions from process equipment. Provides comprehensive
component inventory, leak detection scheduling, emission quantification, and
repair tracking for regulatory compliance.

Key Features:
    - EPA Method 21 leak detection procedures
    - Component inventory management by equipment type
    - Emission factor calculations per EPA AP-42 and TANKS
    - Leak rate quantification using correlation equations
    - Repair tracking and verification
    - LDAR program scheduling and compliance reporting
    - Optical Gas Imaging (OGI) integration support

Regulatory References:
    - 40 CFR Part 60 Subpart VVa (Equipment Leaks VOC)
    - 40 CFR Part 63 Subpart H (Equipment Leaks HAP)
    - EPA Method 21 - Determination of VOC Leaks
    - EPA AP-42 Chapter 5 - Petroleum Industry
    - California CARB LDAR Regulations

Engineering Standards:
    - API 2517 - Evaporative Loss from External Floating Roof Tanks
    - API 2518 - Evaporative Loss from Fixed Roof Tanks

Example:
    >>> ldar = FugitiveEmissionsManager(facility_id="PLANT-001")
    >>> component = ldar.add_component(
    ...     equipment_id="P-101",
    ...     component_type=ComponentType.VALVE,
    ...     service=ServiceType.GAS_VAPOR
    ... )
    >>> inspection = ldar.record_inspection(
    ...     component_id=component.component_id,
    ...     reading_ppm=150
    ... )
    >>> print(f"Leak detected: {inspection.is_leak}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
import math
import statistics
import uuid

from pydantic import BaseModel, Field, validator, root_validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - EPA Method 21 and LDAR Reference Values
# =============================================================================

class LDARConstants:
    """EPA LDAR regulatory constants and thresholds."""

    # Leak definition thresholds (ppm) per 40 CFR 60 Subpart VVa
    LEAK_THRESHOLD_VALVE = 500  # ppm for valves
    LEAK_THRESHOLD_PUMP = 2000  # ppm for pumps
    LEAK_THRESHOLD_COMPRESSOR = 500  # ppm for compressor seals
    LEAK_THRESHOLD_PRESSURE_RELIEF = 500  # ppm for PRDs
    LEAK_THRESHOLD_CONNECTOR = 500  # ppm for connectors
    LEAK_THRESHOLD_OPEN_ENDED = 500  # ppm for open-ended lines

    # California CARB stricter thresholds
    CARB_LEAK_THRESHOLD_VALVE = 100  # ppm
    CARB_LEAK_THRESHOLD_PUMP = 500  # ppm
    CARB_LEAK_THRESHOLD_CONNECTOR = 100  # ppm

    # EPA Method 21 requirements
    PROBE_TRAVERSE_RATE = 1.0  # inch/second max
    SAMPLE_PROBE_DISTANCE = 1.0  # cm from potential leak
    INSTRUMENT_RESPONSE_TIME = 30  # seconds max (T90)
    BACKGROUND_THRESHOLD = 50  # ppm above background is significant

    # Monitoring frequency (days)
    QUARTERLY_MONITORING = 91
    MONTHLY_MONITORING = 30
    WEEKLY_MONITORING = 7

    # Repair timelines (days)
    FIRST_ATTEMPT_REPAIR = 5  # days after detection
    FINAL_REPAIR = 15  # days after detection (or next turnaround)
    DELAY_OF_REPAIR_MAX_DAYS = 365  # maximum delay

    # Emission factors (kg/hr/component) from EPA Protocol
    EMISSION_FACTOR_VALVE_GAS = 0.0089
    EMISSION_FACTOR_VALVE_LIGHT_LIQUID = 0.0024
    EMISSION_FACTOR_VALVE_HEAVY_LIQUID = 0.00003
    EMISSION_FACTOR_PUMP_LIGHT_LIQUID = 0.0199
    EMISSION_FACTOR_PUMP_HEAVY_LIQUID = 0.00862
    EMISSION_FACTOR_COMPRESSOR_SEAL = 0.228
    EMISSION_FACTOR_CONNECTOR_GAS = 0.00083
    EMISSION_FACTOR_CONNECTOR_LIQUID = 0.00025
    EMISSION_FACTOR_OPEN_ENDED = 0.0017
    EMISSION_FACTOR_PRESSURE_RELIEF = 0.104


class ComponentType(Enum):
    """Equipment component types per LDAR regulations."""
    VALVE = "valve"
    PUMP_SEAL = "pump_seal"
    COMPRESSOR_SEAL = "compressor_seal"
    PRESSURE_RELIEF = "pressure_relief"
    CONNECTOR = "connector"
    OPEN_ENDED_LINE = "open_ended_line"
    SAMPLING_CONNECTION = "sampling_connection"
    AGITATOR_SEAL = "agitator_seal"
    FLANGE = "flange"
    INSTRUMENTATION = "instrumentation"


class ServiceType(Enum):
    """Component service type classification."""
    GAS_VAPOR = "gas_vapor"
    LIGHT_LIQUID = "light_liquid"  # VP > 0.3 kPa
    HEAVY_LIQUID = "heavy_liquid"  # VP <= 0.3 kPa
    TWO_PHASE = "two_phase"


class InspectionMethod(Enum):
    """Leak detection inspection methods."""
    EPA_METHOD_21 = "epa_method_21"  # Portable analyzer
    OGI = "ogi"  # Optical Gas Imaging
    SOAP_BUBBLE = "soap_bubble"  # Visual leak test
    ULTRASONIC = "ultrasonic"  # Ultrasonic detection
    AUDIBLE_VISUAL_OLFACTORY = "avo"  # AVO inspection


class LeakStatus(Enum):
    """Component leak status."""
    NO_LEAK = "no_leak"
    LEAK_DETECTED = "leak_detected"
    REPAIR_PENDING = "repair_pending"
    REPAIR_ATTEMPTED = "repair_attempted"
    REPAIRED = "repaired"
    DELAY_OF_REPAIR = "delay_of_repair"
    UNREPAIRABLE = "unrepairable"


class RepairAction(Enum):
    """Repair action types."""
    TIGHTEN_PACKING = "tighten_packing"
    REPLACE_PACKING = "replace_packing"
    REPLACE_GASKET = "replace_gasket"
    REPLACE_SEAL = "replace_seal"
    REPLACE_COMPONENT = "replace_component"
    CAP_OR_PLUG = "cap_or_plug"
    EQUIPMENT_MODIFICATION = "equipment_modification"
    NO_REPAIR_POSSIBLE = "no_repair_possible"


class RegulationProgram(Enum):
    """LDAR regulatory program."""
    EPA_SUBPART_VVA = "epa_subpart_vva"  # 40 CFR 60 Subpart VVa
    EPA_SUBPART_H = "epa_subpart_h"  # 40 CFR 63 Subpart H (HON)
    CARB_LDAR = "carb_ldar"  # California CARB
    TCEQ_28VHP = "tceq_28vhp"  # Texas permit by rule
    BAAQMD = "baaqmd"  # Bay Area AQMD
    CUSTOM = "custom"


# =============================================================================
# DATA MODELS
# =============================================================================

class ComponentInventory(BaseModel):
    """LDAR component inventory entry."""

    component_id: str = Field(..., description="Unique component identifier")
    equipment_id: str = Field(..., description="Parent equipment identifier")
    component_type: ComponentType = Field(..., description="Component type")
    service_type: ServiceType = Field(..., description="Service type")

    # Location
    unit_id: str = Field(..., description="Process unit identifier")
    area: str = Field(default="", description="Plant area")
    location_description: str = Field(default="", description="Physical location")
    p_and_id_reference: Optional[str] = Field(
        default=None, description="P&ID reference"
    )

    # Component details
    size_inches: Optional[float] = Field(
        default=None, ge=0, description="Component size (inches)"
    )
    material: Optional[str] = Field(default=None, description="Component material")
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer")

    # Process information
    stream_composition: Optional[str] = Field(
        default=None, description="Stream composition"
    )
    vapor_pressure_kpa: Optional[float] = Field(
        default=None, ge=0, description="Vapor pressure at operating temp (kPa)"
    )
    operating_temperature_c: Optional[float] = Field(
        default=None, description="Operating temperature (C)"
    )
    operating_pressure_kpa: Optional[float] = Field(
        default=None, description="Operating pressure (kPa)"
    )

    # Regulatory
    regulation_program: RegulationProgram = Field(
        default=RegulationProgram.EPA_SUBPART_VVA,
        description="Applicable LDAR program"
    )
    leak_threshold_ppm: int = Field(
        default=500, ge=0, description="Leak definition threshold (ppm)"
    )
    monitoring_frequency_days: int = Field(
        default=91, ge=1, description="Required monitoring frequency"
    )

    # Status
    current_status: LeakStatus = Field(
        default=LeakStatus.NO_LEAK, description="Current leak status"
    )
    is_accessible: bool = Field(default=True, description="Component accessible")
    is_active: bool = Field(default=True, description="Component in service")
    installation_date: Optional[date] = Field(
        default=None, description="Installation date"
    )

    # Inspection history
    last_inspection_date: Optional[datetime] = Field(
        default=None, description="Last inspection date"
    )
    last_reading_ppm: Optional[float] = Field(
        default=None, description="Last inspection reading (ppm)"
    )
    next_inspection_due: Optional[date] = Field(
        default=None, description="Next inspection due date"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    class Config:
        use_enum_values = True


class InspectionRecord(BaseModel):
    """LDAR inspection record per EPA Method 21."""

    inspection_id: str = Field(..., description="Unique inspection identifier")
    component_id: str = Field(..., description="Component inspected")
    inspection_date: datetime = Field(..., description="Inspection timestamp")

    # Method and readings
    inspection_method: InspectionMethod = Field(..., description="Detection method")
    reading_ppm: float = Field(..., ge=0, description="Instrument reading (ppm)")
    background_ppm: float = Field(
        default=0, ge=0, description="Background reading (ppm)"
    )
    net_reading_ppm: float = Field(default=0, ge=0, description="Net reading (ppm)")

    # Instrument details
    instrument_id: str = Field(default="", description="Instrument identifier")
    instrument_type: str = Field(default="FID", description="Instrument type")
    calibration_date: Optional[date] = Field(
        default=None, description="Last calibration date"
    )
    calibration_gas_ppm: Optional[float] = Field(
        default=None, description="Calibration gas concentration"
    )
    response_factor: float = Field(
        default=1.0, gt=0, description="Response factor for compound"
    )

    # Weather conditions
    ambient_temperature_c: Optional[float] = Field(
        default=None, description="Ambient temperature (C)"
    )
    wind_speed_mph: Optional[float] = Field(
        default=None, ge=0, description="Wind speed (mph)"
    )
    precipitation: bool = Field(default=False, description="Precipitation present")

    # Leak determination
    leak_threshold_ppm: int = Field(..., description="Applicable leak threshold")
    is_leak: bool = Field(..., description="Leak detected")
    leak_magnitude: Optional[str] = Field(
        default=None, description="Leak magnitude classification"
    )

    # Inspector
    inspector_id: str = Field(..., description="Inspector identifier")
    inspector_name: str = Field(default="", description="Inspector name")

    # Notes
    notes: str = Field(default="", description="Inspection notes")
    photo_reference: Optional[str] = Field(
        default=None, description="Photo documentation reference"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    class Config:
        use_enum_values = True


class RepairRecord(BaseModel):
    """Component repair record."""

    repair_id: str = Field(..., description="Unique repair identifier")
    component_id: str = Field(..., description="Component repaired")
    inspection_id: str = Field(..., description="Triggering inspection ID")

    # Repair details
    repair_date: datetime = Field(..., description="Repair timestamp")
    repair_action: RepairAction = Field(..., description="Repair action taken")
    repair_description: str = Field(default="", description="Repair description")

    # Pre/post readings
    pre_repair_reading_ppm: float = Field(
        ..., ge=0, description="Reading before repair (ppm)"
    )
    post_repair_reading_ppm: float = Field(
        ..., ge=0, description="Reading after repair (ppm)"
    )
    repair_successful: bool = Field(..., description="Repair achieved leak-free")

    # Delay of repair
    is_delay_of_repair: bool = Field(
        default=False, description="Delay of repair applied"
    )
    delay_reason: Optional[str] = Field(
        default=None, description="Reason for delay"
    )
    scheduled_repair_date: Optional[date] = Field(
        default=None, description="Scheduled repair date if delayed"
    )

    # Technician
    technician_id: str = Field(..., description="Repair technician ID")
    technician_name: str = Field(default="", description="Technician name")

    # Cost tracking
    parts_cost: Optional[float] = Field(
        default=None, ge=0, description="Parts cost ($)"
    )
    labor_hours: Optional[float] = Field(
        default=None, ge=0, description="Labor hours"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    class Config:
        use_enum_values = True


class EmissionQuantification(BaseModel):
    """Fugitive emission quantification result."""

    component_id: str = Field(..., description="Component identifier")
    calculation_date: datetime = Field(..., description="Calculation timestamp")
    calculation_method: str = Field(..., description="Calculation methodology")

    # Emission rate
    emission_rate_kg_hr: float = Field(
        ..., ge=0, description="Emission rate (kg/hr)"
    )
    emission_rate_lb_hr: float = Field(
        ..., ge=0, description="Emission rate (lb/hr)"
    )
    annual_emissions_tons: float = Field(
        ..., ge=0, description="Annualized emissions (tons/yr)"
    )

    # Methodology details
    screening_value_ppm: Optional[float] = Field(
        default=None, description="Screening value used"
    )
    emission_factor_used: float = Field(..., description="Emission factor applied")
    correlation_equation_used: Optional[str] = Field(
        default=None, description="Correlation equation if applicable"
    )

    # Operating conditions
    operating_hours_per_year: float = Field(
        default=8760, description="Operating hours per year"
    )

    # Uncertainty
    uncertainty_pct: Optional[float] = Field(
        default=None, ge=0, le=100, description="Emission estimate uncertainty (%)"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class FacilityLeakSummary(BaseModel):
    """Facility-wide leak summary."""

    facility_id: str = Field(..., description="Facility identifier")
    reporting_period_start: date = Field(..., description="Period start")
    reporting_period_end: date = Field(..., description="Period end")
    calculation_date: datetime = Field(..., description="Calculation timestamp")

    # Component counts
    total_components: int = Field(default=0, description="Total components")
    components_inspected: int = Field(default=0, description="Components inspected")
    components_leaking: int = Field(default=0, description="Currently leaking")
    components_repaired: int = Field(default=0, description="Repaired this period")

    # Leak rates
    leak_rate_percent: float = Field(
        default=0.0, ge=0, le=100, description="Leak rate (%)"
    )

    # Emissions
    total_fugitive_emissions_tpy: float = Field(
        default=0.0, ge=0, description="Total fugitive emissions (tons/yr)"
    )
    emissions_by_component_type: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by component type"
    )

    # Compliance metrics
    inspection_completion_pct: float = Field(
        default=0.0, ge=0, le=100, description="Inspection completion rate (%)"
    )
    repair_rate_pct: float = Field(
        default=0.0, ge=0, le=100, description="First-attempt repair rate (%)"
    )
    overdue_inspections: int = Field(default=0, description="Overdue inspections")
    delay_of_repair_count: int = Field(default=0, description="Delay of repair count")

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


# =============================================================================
# FUGITIVE EMISSIONS MANAGER
# =============================================================================

class FugitiveEmissionsManager:
    """
    EPA Method 21 Fugitive Emissions and LDAR Program Manager.

    Implements comprehensive fugitive emissions management including:
    - Component inventory tracking
    - EPA Method 21 leak detection scheduling
    - Emission quantification using EPA correlation equations
    - Repair tracking and verification
    - Regulatory compliance reporting

    Regulatory Compliance:
        - 40 CFR 60 Subpart VVa (SOCMI Equipment Leaks)
        - 40 CFR 63 Subpart H (HON Equipment Leaks)
        - California CARB LDAR Regulations
        - API RP 576 Valve Inspection

    Features:
        - Multi-regulation program support
        - Automatic leak threshold application
        - Correlation equation emission calculations
        - Repair timeline tracking
        - OGI integration support
        - Complete audit trail

    Example:
        >>> ldar = FugitiveEmissionsManager(facility_id="PLANT-001")
        >>> component = ldar.add_component(
        ...     equipment_id="P-101",
        ...     component_type=ComponentType.VALVE,
        ...     service=ServiceType.GAS_VAPOR
        ... )
        >>> inspection = ldar.record_inspection(
        ...     component_id=component.component_id,
        ...     reading_ppm=850,
        ...     inspector_id="INSP-001"
        ... )
        >>> if inspection.is_leak:
        ...     ldar.initiate_repair(inspection.inspection_id)
    """

    # EPA correlation equations for emission rate calculation
    # Format: (component_type, service_type): (a, b) where ER = a * (SV)^b
    CORRELATION_EQUATIONS = {
        (ComponentType.VALVE, ServiceType.GAS_VAPOR): (1.87e-6, 0.873),
        (ComponentType.VALVE, ServiceType.LIGHT_LIQUID): (6.41e-6, 0.797),
        (ComponentType.PUMP_SEAL, ServiceType.LIGHT_LIQUID): (1.90e-5, 0.824),
        (ComponentType.COMPRESSOR_SEAL, ServiceType.GAS_VAPOR): (5.03e-5, 0.706),
        (ComponentType.CONNECTOR, ServiceType.GAS_VAPOR): (1.53e-6, 0.735),
        (ComponentType.CONNECTOR, ServiceType.LIGHT_LIQUID): (4.33e-6, 0.663),
        (ComponentType.PRESSURE_RELIEF, ServiceType.GAS_VAPOR): (2.29e-6, 0.746),
    }

    # Default emission factors (kg/hr) for average leak rate
    DEFAULT_EMISSION_FACTORS = {
        (ComponentType.VALVE, ServiceType.GAS_VAPOR): 0.0089,
        (ComponentType.VALVE, ServiceType.LIGHT_LIQUID): 0.0024,
        (ComponentType.VALVE, ServiceType.HEAVY_LIQUID): 0.00003,
        (ComponentType.PUMP_SEAL, ServiceType.LIGHT_LIQUID): 0.0199,
        (ComponentType.PUMP_SEAL, ServiceType.HEAVY_LIQUID): 0.00862,
        (ComponentType.COMPRESSOR_SEAL, ServiceType.GAS_VAPOR): 0.228,
        (ComponentType.CONNECTOR, ServiceType.GAS_VAPOR): 0.00083,
        (ComponentType.CONNECTOR, ServiceType.LIGHT_LIQUID): 0.00025,
        (ComponentType.OPEN_ENDED_LINE, ServiceType.GAS_VAPOR): 0.0017,
        (ComponentType.PRESSURE_RELIEF, ServiceType.GAS_VAPOR): 0.104,
    }

    def __init__(
        self,
        facility_id: str,
        regulation_program: RegulationProgram = RegulationProgram.EPA_SUBPART_VVA,
    ) -> None:
        """
        Initialize Fugitive Emissions Manager.

        Args:
            facility_id: Facility identifier
            regulation_program: Applicable LDAR regulatory program
        """
        self.facility_id = facility_id
        self.regulation_program = regulation_program

        # Component inventory
        self._components: Dict[str, ComponentInventory] = {}

        # Inspection records
        self._inspections: List[InspectionRecord] = []

        # Repair records
        self._repairs: List[RepairRecord] = []

        # Emission calculations
        self._emission_calculations: List[EmissionQuantification] = []

        logger.info(
            f"FugitiveEmissionsManager initialized for {facility_id} "
            f"under {regulation_program.value}"
        )

    # =========================================================================
    # COMPONENT INVENTORY MANAGEMENT
    # =========================================================================

    def add_component(
        self,
        equipment_id: str,
        component_type: ComponentType,
        service_type: ServiceType,
        unit_id: str = "UNIT-001",
        area: str = "",
        location_description: str = "",
        size_inches: Optional[float] = None,
        vapor_pressure_kpa: Optional[float] = None,
        operating_temperature_c: Optional[float] = None,
        custom_leak_threshold: Optional[int] = None,
        custom_monitoring_frequency: Optional[int] = None,
    ) -> ComponentInventory:
        """
        Add component to LDAR inventory.

        Args:
            equipment_id: Parent equipment identifier
            component_type: Component type (valve, pump, etc.)
            service_type: Service type (gas, light liquid, heavy liquid)
            unit_id: Process unit identifier
            area: Plant area
            location_description: Physical location description
            size_inches: Component size in inches
            vapor_pressure_kpa: Stream vapor pressure at operating temp
            operating_temperature_c: Operating temperature
            custom_leak_threshold: Custom leak threshold (ppm)
            custom_monitoring_frequency: Custom monitoring frequency (days)

        Returns:
            Created ComponentInventory object
        """
        component_id = f"{equipment_id}_{component_type.value}_{len(self._components) + 1:04d}"

        # Determine leak threshold based on regulation and component type
        leak_threshold = custom_leak_threshold or self._get_leak_threshold(
            component_type, self.regulation_program
        )

        # Determine monitoring frequency
        monitoring_freq = custom_monitoring_frequency or self._get_monitoring_frequency(
            component_type, self.regulation_program
        )

        # Calculate next inspection due
        next_due = date.today() + timedelta(days=monitoring_freq)

        # Calculate provenance hash
        provenance_hash = self._hash_component_data(
            component_id=component_id,
            equipment_id=equipment_id,
            component_type=component_type.value,
        )

        component = ComponentInventory(
            component_id=component_id,
            equipment_id=equipment_id,
            component_type=component_type,
            service_type=service_type,
            unit_id=unit_id,
            area=area,
            location_description=location_description,
            size_inches=size_inches,
            vapor_pressure_kpa=vapor_pressure_kpa,
            operating_temperature_c=operating_temperature_c,
            regulation_program=self.regulation_program,
            leak_threshold_ppm=leak_threshold,
            monitoring_frequency_days=monitoring_freq,
            next_inspection_due=next_due,
            installation_date=date.today(),
            provenance_hash=provenance_hash,
        )

        self._components[component_id] = component

        logger.info(
            f"Component {component_id} added: {component_type.value} in {service_type.value} service"
        )

        return component

    def _get_leak_threshold(
        self,
        component_type: ComponentType,
        regulation: RegulationProgram,
    ) -> int:
        """Get leak threshold based on component type and regulation."""
        if regulation == RegulationProgram.CARB_LDAR:
            # California CARB stricter thresholds
            thresholds = {
                ComponentType.VALVE: 100,
                ComponentType.PUMP_SEAL: 500,
                ComponentType.COMPRESSOR_SEAL: 100,
                ComponentType.CONNECTOR: 100,
                ComponentType.PRESSURE_RELIEF: 100,
                ComponentType.OPEN_ENDED_LINE: 100,
            }
        else:
            # EPA default thresholds
            thresholds = {
                ComponentType.VALVE: 500,
                ComponentType.PUMP_SEAL: 2000,
                ComponentType.COMPRESSOR_SEAL: 500,
                ComponentType.CONNECTOR: 500,
                ComponentType.PRESSURE_RELIEF: 500,
                ComponentType.OPEN_ENDED_LINE: 500,
            }

        return thresholds.get(component_type, 500)

    def _get_monitoring_frequency(
        self,
        component_type: ComponentType,
        regulation: RegulationProgram,
    ) -> int:
        """Get monitoring frequency in days based on regulation."""
        if regulation == RegulationProgram.CARB_LDAR:
            return 30  # Monthly for CARB
        elif regulation == RegulationProgram.EPA_SUBPART_H:
            # HON has different frequencies
            if component_type == ComponentType.VALVE:
                return 91  # Quarterly
            elif component_type == ComponentType.PUMP_SEAL:
                return 7  # Weekly
        return 91  # Default quarterly

    def get_component(self, component_id: str) -> Optional[ComponentInventory]:
        """Get component by ID."""
        return self._components.get(component_id)

    def get_all_components(
        self,
        component_type: Optional[ComponentType] = None,
        service_type: Optional[ServiceType] = None,
        unit_id: Optional[str] = None,
        leaking_only: bool = False,
    ) -> List[ComponentInventory]:
        """
        Get components with optional filtering.

        Args:
            component_type: Filter by component type
            service_type: Filter by service type
            unit_id: Filter by unit ID
            leaking_only: Return only leaking components

        Returns:
            List of matching ComponentInventory objects
        """
        components = list(self._components.values())

        if component_type:
            components = [c for c in components if c.component_type == component_type.value]
        if service_type:
            components = [c for c in components if c.service_type == service_type.value]
        if unit_id:
            components = [c for c in components if c.unit_id == unit_id]
        if leaking_only:
            components = [c for c in components if c.current_status != LeakStatus.NO_LEAK.value]

        return components

    # =========================================================================
    # INSPECTION MANAGEMENT
    # =========================================================================

    def record_inspection(
        self,
        component_id: str,
        reading_ppm: float,
        inspector_id: str,
        inspection_method: InspectionMethod = InspectionMethod.EPA_METHOD_21,
        background_ppm: float = 0.0,
        instrument_id: str = "",
        calibration_date: Optional[date] = None,
        response_factor: float = 1.0,
        ambient_temperature_c: Optional[float] = None,
        wind_speed_mph: Optional[float] = None,
        inspector_name: str = "",
        notes: str = "",
    ) -> InspectionRecord:
        """
        Record LDAR inspection result per EPA Method 21.

        Args:
            component_id: Component being inspected
            reading_ppm: Instrument reading in ppm
            inspector_id: Inspector identifier
            inspection_method: Detection method used
            background_ppm: Background concentration
            instrument_id: Instrument identifier
            calibration_date: Last calibration date
            response_factor: Compound response factor
            ambient_temperature_c: Ambient temperature
            wind_speed_mph: Wind speed
            inspector_name: Inspector name
            notes: Inspection notes

        Returns:
            InspectionRecord with leak determination

        Raises:
            ValueError: If component not found
        """
        component = self._components.get(component_id)
        if component is None:
            raise ValueError(f"Component not found: {component_id}")

        inspection_id = f"INS_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{len(self._inspections) + 1:06d}"

        # Calculate net reading
        net_reading = max(0, reading_ppm - background_ppm)

        # Apply response factor
        corrected_reading = net_reading * response_factor

        # Determine if leak
        is_leak = corrected_reading >= component.leak_threshold_ppm

        # Classify leak magnitude
        leak_magnitude = None
        if is_leak:
            if corrected_reading >= 10000:
                leak_magnitude = "severe"
            elif corrected_reading >= 5000:
                leak_magnitude = "major"
            elif corrected_reading >= 1000:
                leak_magnitude = "moderate"
            else:
                leak_magnitude = "minor"

        # Calculate provenance hash
        provenance_hash = self._hash_inspection_data(
            inspection_id=inspection_id,
            component_id=component_id,
            reading_ppm=reading_ppm,
        )

        inspection = InspectionRecord(
            inspection_id=inspection_id,
            component_id=component_id,
            inspection_date=datetime.now(timezone.utc),
            inspection_method=inspection_method,
            reading_ppm=reading_ppm,
            background_ppm=background_ppm,
            net_reading_ppm=round(net_reading, 1),
            instrument_id=instrument_id,
            calibration_date=calibration_date,
            response_factor=response_factor,
            ambient_temperature_c=ambient_temperature_c,
            wind_speed_mph=wind_speed_mph,
            leak_threshold_ppm=component.leak_threshold_ppm,
            is_leak=is_leak,
            leak_magnitude=leak_magnitude,
            inspector_id=inspector_id,
            inspector_name=inspector_name,
            notes=notes,
            provenance_hash=provenance_hash,
        )

        self._inspections.append(inspection)

        # Update component
        component.last_inspection_date = inspection.inspection_date
        component.last_reading_ppm = corrected_reading
        component.next_inspection_due = date.today() + timedelta(
            days=component.monitoring_frequency_days
        )

        if is_leak:
            component.current_status = LeakStatus.LEAK_DETECTED
            logger.warning(
                f"LEAK DETECTED at {component_id}: {corrected_reading:.0f} ppm "
                f"(threshold: {component.leak_threshold_ppm} ppm)"
            )
        else:
            if component.current_status == LeakStatus.REPAIRED:
                pass  # Keep repaired status
            else:
                component.current_status = LeakStatus.NO_LEAK

        logger.info(
            f"Inspection {inspection_id} recorded: {reading_ppm:.0f} ppm, "
            f"Leak: {is_leak}"
        )

        return inspection

    def get_overdue_inspections(self) -> List[ComponentInventory]:
        """Get components with overdue inspections."""
        today = date.today()
        overdue = []

        for component in self._components.values():
            if component.is_active and component.is_accessible:
                if component.next_inspection_due and component.next_inspection_due < today:
                    overdue.append(component)

        return sorted(overdue, key=lambda c: c.next_inspection_due)

    def get_inspection_schedule(
        self,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Get inspection schedule for a date range.

        Args:
            start_date: Schedule start date
            end_date: Schedule end date

        Returns:
            List of scheduled inspections
        """
        schedule = []

        for component in self._components.values():
            if not component.is_active or not component.is_accessible:
                continue

            if component.next_inspection_due:
                if start_date <= component.next_inspection_due <= end_date:
                    schedule.append({
                        "component_id": component.component_id,
                        "equipment_id": component.equipment_id,
                        "component_type": component.component_type,
                        "unit_id": component.unit_id,
                        "area": component.area,
                        "due_date": component.next_inspection_due.isoformat(),
                        "priority": self._calculate_inspection_priority(component),
                    })

        return sorted(schedule, key=lambda x: x["due_date"])

    def _calculate_inspection_priority(self, component: ComponentInventory) -> str:
        """Calculate inspection priority based on history and component type."""
        # Higher priority for previously leaking components
        if component.current_status != LeakStatus.NO_LEAK.value:
            return "high"

        # Higher priority for pumps and compressors
        if component.component_type in [
            ComponentType.PUMP_SEAL.value,
            ComponentType.COMPRESSOR_SEAL.value,
        ]:
            return "medium"

        # Check if overdue
        if component.next_inspection_due and component.next_inspection_due < date.today():
            return "high"

        return "normal"

    # =========================================================================
    # REPAIR MANAGEMENT
    # =========================================================================

    def record_repair(
        self,
        component_id: str,
        inspection_id: str,
        repair_action: RepairAction,
        technician_id: str,
        post_repair_reading_ppm: float,
        repair_description: str = "",
        technician_name: str = "",
        is_delay_of_repair: bool = False,
        delay_reason: Optional[str] = None,
        scheduled_repair_date: Optional[date] = None,
        parts_cost: Optional[float] = None,
        labor_hours: Optional[float] = None,
    ) -> RepairRecord:
        """
        Record component repair.

        Args:
            component_id: Component repaired
            inspection_id: Inspection that triggered repair
            repair_action: Type of repair performed
            technician_id: Repair technician ID
            post_repair_reading_ppm: Reading after repair
            repair_description: Description of repair work
            technician_name: Technician name
            is_delay_of_repair: Delay of repair applied
            delay_reason: Reason for delay
            scheduled_repair_date: Future repair date if delayed
            parts_cost: Cost of parts
            labor_hours: Labor hours

        Returns:
            RepairRecord
        """
        component = self._components.get(component_id)
        if component is None:
            raise ValueError(f"Component not found: {component_id}")

        # Find triggering inspection
        inspection = next(
            (i for i in self._inspections if i.inspection_id == inspection_id),
            None
        )
        pre_repair_reading = inspection.reading_ppm if inspection else 0

        repair_id = f"REP_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{len(self._repairs) + 1:04d}"

        # Determine if repair was successful
        repair_successful = post_repair_reading_ppm < component.leak_threshold_ppm

        # Calculate provenance hash
        provenance_hash = self._hash_repair_data(
            repair_id=repair_id,
            component_id=component_id,
            post_reading=post_repair_reading_ppm,
        )

        repair = RepairRecord(
            repair_id=repair_id,
            component_id=component_id,
            inspection_id=inspection_id,
            repair_date=datetime.now(timezone.utc),
            repair_action=repair_action,
            repair_description=repair_description,
            pre_repair_reading_ppm=pre_repair_reading,
            post_repair_reading_ppm=post_repair_reading_ppm,
            repair_successful=repair_successful,
            is_delay_of_repair=is_delay_of_repair,
            delay_reason=delay_reason,
            scheduled_repair_date=scheduled_repair_date,
            technician_id=technician_id,
            technician_name=technician_name,
            parts_cost=parts_cost,
            labor_hours=labor_hours,
            provenance_hash=provenance_hash,
        )

        self._repairs.append(repair)

        # Update component status
        if is_delay_of_repair:
            component.current_status = LeakStatus.DELAY_OF_REPAIR
        elif repair_successful:
            component.current_status = LeakStatus.REPAIRED
        else:
            component.current_status = LeakStatus.REPAIR_ATTEMPTED

        logger.info(
            f"Repair {repair_id} recorded for {component_id}: "
            f"{'Successful' if repair_successful else 'Unsuccessful'} "
            f"({post_repair_reading_ppm:.0f} ppm)"
        )

        return repair

    def get_repairs_due(self) -> List[Dict[str, Any]]:
        """Get components requiring repair within regulatory timeline."""
        repairs_due = []

        for component in self._components.values():
            if component.current_status in [
                LeakStatus.LEAK_DETECTED.value,
                LeakStatus.REPAIR_ATTEMPTED.value,
            ]:
                # Find detection date
                leak_inspection = self._get_last_leak_inspection(component.component_id)
                if leak_inspection:
                    detection_date = leak_inspection.inspection_date.date()
                    first_attempt_due = detection_date + timedelta(
                        days=LDARConstants.FIRST_ATTEMPT_REPAIR
                    )
                    final_repair_due = detection_date + timedelta(
                        days=LDARConstants.FINAL_REPAIR
                    )

                    repairs_due.append({
                        "component_id": component.component_id,
                        "equipment_id": component.equipment_id,
                        "detection_date": detection_date.isoformat(),
                        "reading_ppm": leak_inspection.reading_ppm,
                        "first_attempt_due": first_attempt_due.isoformat(),
                        "final_repair_due": final_repair_due.isoformat(),
                        "days_since_detection": (date.today() - detection_date).days,
                        "priority": "high" if date.today() > first_attempt_due else "normal",
                    })

        return sorted(repairs_due, key=lambda x: x["detection_date"])

    def _get_last_leak_inspection(
        self, component_id: str
    ) -> Optional[InspectionRecord]:
        """Get the most recent leak inspection for a component."""
        leak_inspections = [
            i for i in self._inspections
            if i.component_id == component_id and i.is_leak
        ]
        return leak_inspections[-1] if leak_inspections else None

    # =========================================================================
    # EMISSION QUANTIFICATION
    # =========================================================================

    def calculate_component_emissions(
        self,
        component_id: str,
        screening_value_ppm: Optional[float] = None,
        use_correlation_equation: bool = True,
        operating_hours_per_year: float = 8760,
    ) -> EmissionQuantification:
        """
        Calculate fugitive emissions for a component.

        Uses EPA correlation equations or emission factors per
        EPA Protocol for Equipment Leak Emission Estimates.

        Args:
            component_id: Component identifier
            screening_value_ppm: Screening value (uses last reading if None)
            use_correlation_equation: Use correlation equation vs emission factor
            operating_hours_per_year: Annual operating hours

        Returns:
            EmissionQuantification with emission rate

        Raises:
            ValueError: If component not found
        """
        component = self._components.get(component_id)
        if component is None:
            raise ValueError(f"Component not found: {component_id}")

        # Get screening value
        if screening_value_ppm is None:
            screening_value_ppm = component.last_reading_ppm or 0

        # Get component type and service type
        comp_type = ComponentType(component.component_type)
        svc_type = ServiceType(component.service_type)

        emission_rate_kg_hr = 0.0
        calculation_method = ""
        correlation_used = None
        emission_factor = 0.0

        if use_correlation_equation and screening_value_ppm > 0:
            # Try correlation equation
            key = (comp_type, svc_type)
            if key in self.CORRELATION_EQUATIONS:
                a, b = self.CORRELATION_EQUATIONS[key]
                emission_rate_kg_hr = a * (screening_value_ppm ** b)
                calculation_method = "EPA Correlation Equation"
                correlation_used = f"ER = {a:.2e} * SV^{b:.3f}"
            else:
                # Fall back to emission factor
                emission_factor = self.DEFAULT_EMISSION_FACTORS.get(key, 0.001)
                emission_rate_kg_hr = emission_factor
                calculation_method = "EPA Average Emission Factor"
        else:
            # Use emission factor
            key = (comp_type, svc_type)
            emission_factor = self.DEFAULT_EMISSION_FACTORS.get(key, 0.001)

            # If leaking, use leaker factor; otherwise, non-leaker factor
            if screening_value_ppm and screening_value_ppm >= component.leak_threshold_ppm:
                emission_rate_kg_hr = emission_factor  # Leaker factor
            else:
                # Non-leaker factor is typically 10-100x lower
                emission_rate_kg_hr = emission_factor * 0.1

            calculation_method = "EPA Emission Factor"

        # Convert to lb/hr
        emission_rate_lb_hr = emission_rate_kg_hr * 2.205

        # Annualize
        annual_emissions_tons = (
            emission_rate_lb_hr * operating_hours_per_year / 2000
        )

        # Calculate provenance hash
        provenance_hash = hashlib.sha256(
            f"{component_id}:{emission_rate_kg_hr}:{screening_value_ppm}:"
            f"{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()

        quantification = EmissionQuantification(
            component_id=component_id,
            calculation_date=datetime.now(timezone.utc),
            calculation_method=calculation_method,
            emission_rate_kg_hr=round(emission_rate_kg_hr, 6),
            emission_rate_lb_hr=round(emission_rate_lb_hr, 6),
            annual_emissions_tons=round(annual_emissions_tons, 4),
            screening_value_ppm=screening_value_ppm,
            emission_factor_used=emission_factor,
            correlation_equation_used=correlation_used,
            operating_hours_per_year=operating_hours_per_year,
            uncertainty_pct=30.0,  # EPA guidance
            provenance_hash=provenance_hash,
        )

        self._emission_calculations.append(quantification)

        logger.debug(
            f"Emissions calculated for {component_id}: "
            f"{emission_rate_lb_hr:.6f} lb/hr, {annual_emissions_tons:.4f} tons/yr"
        )

        return quantification

    def calculate_facility_emissions(
        self,
        unit_id: Optional[str] = None,
        operating_hours_per_year: float = 8760,
    ) -> Dict[str, Any]:
        """
        Calculate total fugitive emissions for facility or unit.

        Args:
            unit_id: Specific unit (None for entire facility)
            operating_hours_per_year: Annual operating hours

        Returns:
            Dictionary with total emissions by category
        """
        components = self.get_all_components(unit_id=unit_id)

        total_kg_hr = 0.0
        by_component_type: Dict[str, float] = {}
        by_service_type: Dict[str, float] = {}

        for component in components:
            if not component.is_active:
                continue

            quant = self.calculate_component_emissions(
                component_id=component.component_id,
                operating_hours_per_year=operating_hours_per_year,
            )

            total_kg_hr += quant.emission_rate_kg_hr

            # Aggregate by component type
            comp_type = component.component_type
            by_component_type[comp_type] = (
                by_component_type.get(comp_type, 0) + quant.annual_emissions_tons
            )

            # Aggregate by service type
            svc_type = component.service_type
            by_service_type[svc_type] = (
                by_service_type.get(svc_type, 0) + quant.annual_emissions_tons
            )

        total_lb_hr = total_kg_hr * 2.205
        total_tons_yr = total_lb_hr * operating_hours_per_year / 2000

        return {
            "facility_id": self.facility_id,
            "unit_id": unit_id or "ALL",
            "calculation_date": datetime.now(timezone.utc).isoformat(),
            "total_components": len(components),
            "total_emission_rate_kg_hr": round(total_kg_hr, 4),
            "total_emission_rate_lb_hr": round(total_lb_hr, 4),
            "total_annual_emissions_tons": round(total_tons_yr, 2),
            "by_component_type": by_component_type,
            "by_service_type": by_service_type,
            "operating_hours": operating_hours_per_year,
        }

    # =========================================================================
    # COMPLIANCE REPORTING
    # =========================================================================

    def generate_leak_summary(
        self,
        start_date: date,
        end_date: date,
    ) -> FacilityLeakSummary:
        """
        Generate LDAR compliance summary for reporting period.

        Args:
            start_date: Reporting period start
            end_date: Reporting period end

        Returns:
            FacilityLeakSummary with compliance metrics
        """
        # Get period inspections
        period_inspections = [
            i for i in self._inspections
            if start_date <= i.inspection_date.date() <= end_date
        ]

        # Get period repairs
        period_repairs = [
            r for r in self._repairs
            if start_date <= r.repair_date.date() <= end_date
        ]

        # Count components
        total_components = len([c for c in self._components.values() if c.is_active])
        components_inspected = len(set(i.component_id for i in period_inspections))

        # Count leaks
        leak_inspections = [i for i in period_inspections if i.is_leak]
        components_leaking = len(set(i.component_id for i in leak_inspections))

        # Count repairs
        successful_repairs = [r for r in period_repairs if r.repair_successful]
        components_repaired = len(successful_repairs)

        # Calculate leak rate
        leak_rate = (
            components_leaking / components_inspected * 100
            if components_inspected > 0 else 0
        )

        # Calculate emissions
        emissions_result = self.calculate_facility_emissions()
        total_emissions_tpy = emissions_result["total_annual_emissions_tons"]

        # Inspection completion
        inspection_completion = (
            components_inspected / total_components * 100
            if total_components > 0 else 0
        )

        # Repair rate
        first_attempt_success = len([r for r in period_repairs if r.repair_successful])
        repair_rate = (
            first_attempt_success / len(period_repairs) * 100
            if period_repairs else 100
        )

        # Overdue inspections
        overdue = len(self.get_overdue_inspections())

        # Delay of repair count
        dor_count = len([
            r for r in period_repairs if r.is_delay_of_repair
        ])

        # Provenance hash
        provenance_hash = hashlib.sha256(
            f"{self.facility_id}:{start_date}:{end_date}:"
            f"{total_emissions_tpy}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()

        summary = FacilityLeakSummary(
            facility_id=self.facility_id,
            reporting_period_start=start_date,
            reporting_period_end=end_date,
            calculation_date=datetime.now(timezone.utc),
            total_components=total_components,
            components_inspected=components_inspected,
            components_leaking=components_leaking,
            components_repaired=components_repaired,
            leak_rate_percent=round(leak_rate, 2),
            total_fugitive_emissions_tpy=round(total_emissions_tpy, 2),
            emissions_by_component_type=emissions_result["by_component_type"],
            inspection_completion_pct=round(inspection_completion, 1),
            repair_rate_pct=round(repair_rate, 1),
            overdue_inspections=overdue,
            delay_of_repair_count=dor_count,
            provenance_hash=provenance_hash,
        )

        logger.info(
            f"Leak summary generated: {components_inspected} inspected, "
            f"{components_leaking} leaking ({leak_rate:.1f}%), "
            f"{total_emissions_tpy:.2f} tons/yr"
        )

        return summary

    def export_component_inventory(self) -> List[Dict[str, Any]]:
        """Export component inventory for regulatory reporting."""
        inventory = []

        for component in self._components.values():
            inventory.append({
                "component_id": component.component_id,
                "equipment_id": component.equipment_id,
                "component_type": component.component_type,
                "service_type": component.service_type,
                "unit_id": component.unit_id,
                "area": component.area,
                "location": component.location_description,
                "size_inches": component.size_inches,
                "leak_threshold_ppm": component.leak_threshold_ppm,
                "monitoring_frequency_days": component.monitoring_frequency_days,
                "current_status": component.current_status,
                "last_inspection_date": (
                    component.last_inspection_date.isoformat()
                    if component.last_inspection_date else None
                ),
                "last_reading_ppm": component.last_reading_ppm,
                "next_inspection_due": (
                    component.next_inspection_due.isoformat()
                    if component.next_inspection_due else None
                ),
                "is_active": component.is_active,
            })

        return inventory

    def export_inspection_history(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Export inspection history for regulatory reporting."""
        inspections = self._inspections

        if start_date:
            inspections = [
                i for i in inspections if i.inspection_date.date() >= start_date
            ]
        if end_date:
            inspections = [
                i for i in inspections if i.inspection_date.date() <= end_date
            ]

        return [
            {
                "inspection_id": i.inspection_id,
                "component_id": i.component_id,
                "inspection_date": i.inspection_date.isoformat(),
                "method": i.inspection_method,
                "reading_ppm": i.reading_ppm,
                "background_ppm": i.background_ppm,
                "net_reading_ppm": i.net_reading_ppm,
                "leak_threshold_ppm": i.leak_threshold_ppm,
                "is_leak": i.is_leak,
                "leak_magnitude": i.leak_magnitude,
                "instrument_id": i.instrument_id,
                "inspector_id": i.inspector_id,
            }
            for i in inspections
        ]

    # =========================================================================
    # HASH UTILITIES
    # =========================================================================

    def _hash_component_data(
        self,
        component_id: str,
        equipment_id: str,
        component_type: str,
    ) -> str:
        """Calculate SHA-256 hash for component provenance."""
        data = {
            "component_id": component_id,
            "equipment_id": equipment_id,
            "component_type": component_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def _hash_inspection_data(
        self,
        inspection_id: str,
        component_id: str,
        reading_ppm: float,
    ) -> str:
        """Calculate SHA-256 hash for inspection provenance."""
        data = {
            "inspection_id": inspection_id,
            "component_id": component_id,
            "reading_ppm": reading_ppm,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def _hash_repair_data(
        self,
        repair_id: str,
        component_id: str,
        post_reading: float,
    ) -> str:
        """Calculate SHA-256 hash for repair provenance."""
        data = {
            "repair_id": repair_id,
            "component_id": component_id,
            "post_reading": post_reading,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall LDAR compliance status."""
        today = date.today()

        overdue = self.get_overdue_inspections()
        repairs_due = self.get_repairs_due()

        leaking_components = len([
            c for c in self._components.values()
            if c.current_status in [
                LeakStatus.LEAK_DETECTED.value,
                LeakStatus.REPAIR_ATTEMPTED.value,
            ]
        ])

        return {
            "facility_id": self.facility_id,
            "regulation_program": self.regulation_program.value,
            "as_of_date": today.isoformat(),
            "total_components": len(self._components),
            "active_components": len([
                c for c in self._components.values() if c.is_active
            ]),
            "overdue_inspections": len(overdue),
            "leaking_components": leaking_components,
            "repairs_due": len(repairs_due),
            "compliance_status": (
                "COMPLIANT" if len(overdue) == 0 and leaking_components == 0
                else "NON-COMPLIANT"
            ),
            "next_inspection_due": (
                min(
                    (c.next_inspection_due for c in self._components.values()
                     if c.next_inspection_due and c.is_active),
                    default=None
                )
            ),
        }
