# -*- coding: utf-8 -*-
"""
Data Transformation Utilities for GL-001 ProcessHeatOrchestrator

Provides comprehensive data transformation, normalization, and validation for:
- SCADA sensor data normalization
- ERP data parsing and enrichment
- Agent message formatting
- Unit conversions
- Data quality validation

Features:
- Multi-format data parsing
- Unit conversion library
- Data validation and cleansing
- Schema enforcement
- Performance optimization for high-volume data
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import math
from greenlang.determinism import DeterministicClock
from greenlang.determinism import FinancialDecimal
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = 5  # 95-100% quality
    GOOD = 4       # 80-95% quality
    FAIR = 3       # 60-80% quality
    POOR = 2       # 40-60% quality
    INVALID = 1    # <40% quality


class UnitType(Enum):
    """Types of measurement units."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    ENERGY = "energy"
    POWER = "power"
    VOLUME = "volume"
    MASS = "mass"
    TIME = "time"


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    quality_score: float  # 0-100
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrections: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformationMetrics:
    """Metrics for data transformation performance."""
    records_processed: int = 0
    records_valid: int = 0
    records_invalid: int = 0
    processing_time_ms: float = 0
    average_quality_score: float = 0


class UnitConverter:
    """
    Universal unit converter for industrial measurements.

    Handles all common industrial units with high precision.
    """

    def __init__(self):
        """Initialize unit converter with conversion factors."""
        self.conversions = {
            UnitType.TEMPERATURE: {
                ('celsius', 'fahrenheit'): lambda x: x * 9/5 + 32,
                ('fahrenheit', 'celsius'): lambda x: (x - 32) * 5/9,
                ('celsius', 'kelvin'): lambda x: x + 273.15,
                ('kelvin', 'celsius'): lambda x: x - 273.15,
                ('fahrenheit', 'kelvin'): lambda x: (x - 32) * 5/9 + 273.15,
                ('kelvin', 'fahrenheit'): lambda x: (x - 273.15) * 9/5 + 32
            },
            UnitType.PRESSURE: {
                ('bar', 'psi'): lambda x: x * 14.5038,
                ('psi', 'bar'): lambda x: x / 14.5038,
                ('bar', 'kpa'): lambda x: x * 100,
                ('kpa', 'bar'): lambda x: x / 100,
                ('psi', 'kpa'): lambda x: x * 6.89476,
                ('kpa', 'psi'): lambda x: x / 6.89476,
                ('bar', 'atm'): lambda x: x / 1.01325,
                ('atm', 'bar'): lambda x: x * 1.01325
            },
            UnitType.FLOW: {
                ('m3/h', 'l/min'): lambda x: x * 16.6667,
                ('l/min', 'm3/h'): lambda x: x / 16.6667,
                ('m3/h', 'gpm'): lambda x: x * 4.40287,
                ('gpm', 'm3/h'): lambda x: x / 4.40287,
                ('l/min', 'gpm'): lambda x: x * 0.264172,
                ('gpm', 'l/min'): lambda x: x / 0.264172,
                ('m3/h', 'cfm'): lambda x: x * 0.588578,
                ('cfm', 'm3/h'): lambda x: x / 0.588578
            },
            UnitType.ENERGY: {
                ('kwh', 'mwh'): lambda x: x / 1000,
                ('mwh', 'kwh'): lambda x: x * 1000,
                ('kwh', 'btu'): lambda x: x * 3412.14,
                ('btu', 'kwh'): lambda x: x / 3412.14,
                ('kwh', 'mj'): lambda x: x * 3.6,
                ('mj', 'kwh'): lambda x: x / 3.6,
                ('btu', 'mj'): lambda x: x * 0.00105506,
                ('mj', 'btu'): lambda x: x / 0.00105506
            },
            UnitType.POWER: {
                ('kw', 'mw'): lambda x: x / 1000,
                ('mw', 'kw'): lambda x: x * 1000,
                ('kw', 'hp'): lambda x: x * 1.34102,
                ('hp', 'kw'): lambda x: x / 1.34102,
                ('kw', 'btu/h'): lambda x: x * 3412.14,
                ('btu/h', 'kw'): lambda x: x / 3412.14
            },
            UnitType.MASS: {
                ('kg', 'lb'): lambda x: x * 2.20462,
                ('lb', 'kg'): lambda x: x / 2.20462,
                ('kg', 'ton'): lambda x: x / 1000,
                ('ton', 'kg'): lambda x: x * 1000,
                ('lb', 'ton'): lambda x: x / 2204.62,
                ('ton', 'lb'): lambda x: x * 2204.62
            }
        }

    def convert(self, value: float, from_unit: str, to_unit: str, unit_type: UnitType) -> float:
        """
        Convert value between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            unit_type: Type of unit

        Returns:
            Converted value

        Raises:
            ValueError: If conversion not supported
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        if from_unit == to_unit:
            return value

        if unit_type not in self.conversions:
            raise ValueError(f"Unit type {unit_type} not supported")

        conversion_key = (from_unit, to_unit)
        if conversion_key in self.conversions[unit_type]:
            return self.conversions[unit_type][conversion_key](value)

        # Try reverse conversion
        reverse_key = (to_unit, from_unit)
        if reverse_key in self.conversions[unit_type]:
            return 1.0 / self.conversions[unit_type][reverse_key](1.0 / value)

        raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported for {unit_type}")


class DataValidator:
    """
    Validates data quality and completeness.

    Implements comprehensive validation rules for industrial data.
    """

    def __init__(self):
        """Initialize data validator."""
        self.unit_converter = UnitConverter()

    def validate_sensor_data(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate SCADA sensor data.

        Args:
            data: Sensor data dictionary

        Returns:
            Validation result with quality score
        """
        errors = []
        warnings = []
        corrections = {}
        quality_points = 100

        # Required fields
        required_fields = ['sensor_id', 'value', 'timestamp']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                quality_points -= 20

        # Validate sensor_id format
        if 'sensor_id' in data:
            if not re.match(r'^[A-Z]+_\d{3}$', data['sensor_id']):
                warnings.append(f"Non-standard sensor_id format: {data['sensor_id']}")
                quality_points -= 5

        # Validate value
        if 'value' in data:
            try:
                value = float(data['value'])

                # Check for out-of-range values
                if 'min_value' in data and value < data['min_value']:
                    warnings.append(f"Value {value} below minimum {data['min_value']}")
                    quality_points -= 10

                if 'max_value' in data and value > data['max_value']:
                    warnings.append(f"Value {value} above maximum {data['max_value']}")
                    quality_points -= 10

                # Check for NaN or Inf
                if math.isnan(value) or math.isinf(value):
                    errors.append(f"Invalid numeric value: {value}")
                    quality_points -= 30

            except (TypeError, ValueError):
                errors.append(f"Invalid value type: {data['value']}")
                quality_points -= 25

        # Validate timestamp
        if 'timestamp' in data:
            try:
                if isinstance(data['timestamp'], str):
                    datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                elif not isinstance(data['timestamp'], datetime):
                    warnings.append(f"Non-standard timestamp format")
                    quality_points -= 5
            except ValueError:
                errors.append(f"Invalid timestamp format: {data['timestamp']}")
                quality_points -= 15

        # Validate unit
        if 'unit' in data:
            valid_units = ['celsius', 'fahrenheit', 'kelvin', 'bar', 'psi', 'kpa',
                          'm3/h', 'l/min', 'gpm', 'kwh', 'mwh', 'kw', 'mw']
            if data['unit'].lower() not in valid_units:
                warnings.append(f"Unknown unit: {data['unit']}")
                quality_points -= 5

        # Calculate final quality score
        quality_score = max(0, min(100, quality_points))

        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=quality_score,
            errors=errors,
            warnings=warnings,
            corrections=corrections
        )

    def validate_erp_data(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate ERP data.

        Args:
            data: ERP data dictionary

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        corrections = {}
        quality_points = 100

        # Check for required ERP fields
        required_fields = ['transaction_id', 'date', 'amount']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                quality_points -= 15

        # Validate transaction_id
        if 'transaction_id' in data:
            if not data['transaction_id']:
                errors.append("Empty transaction_id")
                quality_points -= 20

        # Validate date
        if 'date' in data:
            try:
                datetime.fromisoformat(str(data['date']))
            except ValueError:
                errors.append(f"Invalid date format: {data['date']}")
                quality_points -= 15

        # Validate amount
        if 'amount' in data:
            try:
                amount = FinancialDecimal.from_string(data['amount'])
                if amount < 0:
                    warnings.append(f"Negative amount: {amount}")
                    quality_points -= 10
            except (TypeError, ValueError):
                errors.append(f"Invalid amount: {data['amount']}")
                quality_points -= 20

        # Validate currency
        if 'currency' in data:
            valid_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY']
            if data['currency'] not in valid_currencies:
                warnings.append(f"Unknown currency: {data['currency']}")
                quality_points -= 5

        quality_score = max(0, min(100, quality_points))

        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=quality_score,
            errors=errors,
            warnings=warnings,
            corrections=corrections
        )


class SCADADataTransformer:
    """
    Transforms SCADA data into standardized format.

    Handles sensor data normalization, calibration, and enrichment.
    """

    def __init__(self):
        """Initialize SCADA data transformer."""
        self.validator = DataValidator()
        self.unit_converter = UnitConverter()
        self.metrics = TransformationMetrics()

    def transform_sensor_reading(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform single sensor reading.

        Args:
            raw_data: Raw sensor data

        Returns:
            Transformed sensor data
        """
        # Validate input
        validation = self.validator.validate_sensor_data(raw_data)

        # Start with raw data
        transformed = raw_data.copy()

        # Normalize sensor ID
        if 'sensor_id' in transformed:
            transformed['sensor_id'] = transformed['sensor_id'].upper()

        # Apply calibration factor
        if 'value' in transformed and 'calibration_factor' in transformed:
            try:
                transformed['calibrated_value'] = FinancialDecimal.from_string(transformed['value']) * FinancialDecimal.from_string(transformed['calibration_factor'])
            except (TypeError, ValueError):
                logger.warning(f"Failed to apply calibration factor for sensor {transformed.get('sensor_id')}")

        # Normalize timestamp
        if 'timestamp' in transformed:
            if isinstance(transformed['timestamp'], str):
                try:
                    transformed['timestamp'] = datetime.fromisoformat(
                        transformed['timestamp'].replace('Z', '+00:00')
                    )
                except ValueError:
                    transformed['timestamp'] = DeterministicClock.utcnow()
            elif not isinstance(transformed['timestamp'], datetime):
                transformed['timestamp'] = DeterministicClock.utcnow()

        # Add metadata
        transformed['quality_score'] = validation.quality_score
        transformed['quality_level'] = self._get_quality_level(validation.quality_score)
        transformed['validated'] = validation.is_valid
        transformed['transformation_timestamp'] = DeterministicClock.utcnow().isoformat()

        # Update metrics
        self.metrics.records_processed += 1
        if validation.is_valid:
            self.metrics.records_valid += 1
        else:
            self.metrics.records_invalid += 1

        return transformed

    def transform_batch(self, raw_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform batch of sensor readings.

        Args:
            raw_data_list: List of raw sensor data

        Returns:
            List of transformed sensor data
        """
        start_time = DeterministicClock.utcnow()
        transformed_list = []

        for raw_data in raw_data_list:
            transformed = self.transform_sensor_reading(raw_data)
            transformed_list.append(transformed)

        # Calculate metrics
        processing_time = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
        self.metrics.processing_time_ms = processing_time

        if transformed_list:
            total_quality = sum(t.get('quality_score', 0) for t in transformed_list)
            self.metrics.average_quality_score = total_quality / len(transformed_list)

        return transformed_list

    def normalize_units(self, data: Dict[str, Any], target_unit: str) -> Dict[str, Any]:
        """
        Normalize data to target unit.

        Args:
            data: Data with unit information
            target_unit: Target unit to convert to

        Returns:
            Data with normalized unit
        """
        if 'value' not in data or 'unit' not in data:
            return data

        normalized = data.copy()

        try:
            # Determine unit type
            unit_type = self._detect_unit_type(data['unit'])

            if unit_type:
                # Convert to target unit
                converted_value = self.unit_converter.convert(
                    float(data['value']),
                    data['unit'],
                    target_unit,
                    unit_type
                )

                normalized['original_value'] = data['value']
                normalized['original_unit'] = data['unit']
                normalized['value'] = converted_value
                normalized['unit'] = target_unit

        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to normalize units: {e}")

        return normalized

    def _get_quality_level(self, score: float) -> str:
        """Get quality level from score."""
        if score >= 95:
            return DataQuality.EXCELLENT.name
        elif score >= 80:
            return DataQuality.GOOD.name
        elif score >= 60:
            return DataQuality.FAIR.name
        elif score >= 40:
            return DataQuality.POOR.name
        else:
            return DataQuality.INVALID.name

    def _detect_unit_type(self, unit: str) -> Optional[UnitType]:
        """Detect unit type from unit string."""
        unit_lower = unit.lower()

        temperature_units = ['celsius', 'fahrenheit', 'kelvin', 'c', 'f', 'k']
        pressure_units = ['bar', 'psi', 'kpa', 'atm', 'pascal']
        flow_units = ['m3/h', 'l/min', 'gpm', 'cfm']
        energy_units = ['kwh', 'mwh', 'btu', 'mj', 'gj']
        power_units = ['kw', 'mw', 'hp', 'btu/h']

        if unit_lower in temperature_units:
            return UnitType.TEMPERATURE
        elif unit_lower in pressure_units:
            return UnitType.PRESSURE
        elif unit_lower in flow_units:
            return UnitType.FLOW
        elif unit_lower in energy_units:
            return UnitType.ENERGY
        elif unit_lower in power_units:
            return UnitType.POWER

        return None


class ERPDataTransformer:
    """
    Transforms ERP data into standardized format.

    Handles ERP-specific data parsing, enrichment, and normalization.
    """

    def __init__(self):
        """Initialize ERP data transformer."""
        self.validator = DataValidator()
        self.metrics = TransformationMetrics()

    def transform_energy_consumption(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform energy consumption data from ERP.

        Args:
            raw_data: Raw ERP energy data

        Returns:
            Transformed energy consumption data
        """
        # Validate input
        validation = self.validator.validate_erp_data(raw_data)

        transformed = {
            'record_type': 'energy_consumption',
            'source_system': raw_data.get('source_system', 'ERP'),
            'transformation_timestamp': DeterministicClock.utcnow().isoformat()
        }

        # Map ERP fields to standard fields
        field_mapping = {
            'PlantCode': 'plant_code',
            'Date': 'date',
            'EnergyType': 'energy_type',
            'Consumption': 'consumption_value',
            'Unit': 'consumption_unit',
            'Cost': 'cost_amount',
            'Currency': 'cost_currency',
            'LocationCode': 'location_code',
            'TransactionDate': 'transaction_date',
            'Quantity': 'quantity',
            'UOM': 'unit_of_measure',
            'TotalCost': 'total_cost'
        }

        for erp_field, standard_field in field_mapping.items():
            if erp_field in raw_data:
                transformed[standard_field] = raw_data[erp_field]

        # Normalize date format
        date_fields = ['date', 'transaction_date']
        for field in date_fields:
            if field in transformed:
                transformed[field] = self._normalize_date(transformed[field])

        # Calculate derived fields
        if 'consumption_value' in transformed and 'cost_amount' in transformed:
            try:
                consumption = FinancialDecimal.from_string(transformed['consumption_value'])
                cost = FinancialDecimal.from_string(transformed['cost_amount'])
                if consumption > 0:
                    transformed['unit_cost'] = cost / consumption
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        # Add quality metadata
        transformed['quality_score'] = validation.quality_score
        transformed['validated'] = validation.is_valid
        if not validation.is_valid:
            transformed['validation_errors'] = validation.errors

        # Update metrics
        self.metrics.records_processed += 1
        if validation.is_valid:
            self.metrics.records_valid += 1
        else:
            self.metrics.records_invalid += 1

        return transformed

    def transform_production_schedule(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform production schedule data from ERP.

        Args:
            raw_data: Raw production schedule data

        Returns:
            Transformed production schedule
        """
        transformed = {
            'record_type': 'production_schedule',
            'source_system': raw_data.get('source_system', 'ERP'),
            'transformation_timestamp': DeterministicClock.utcnow().isoformat()
        }

        # Map fields
        field_mapping = {
            'OrderNumber': 'order_number',
            'Product': 'product_code',
            'PlantCode': 'plant_code',
            'PlannedStart': 'planned_start_date',
            'PlannedEnd': 'planned_end_date',
            'Quantity': 'planned_quantity',
            'Unit': 'quantity_unit',
            'Status': 'order_status',
            'WorkOrderNumber': 'work_order_number',
            'AssetNumber': 'asset_number',
            'MaintenanceType': 'maintenance_type'
        }

        for erp_field, standard_field in field_mapping.items():
            if erp_field in raw_data:
                transformed[standard_field] = raw_data[erp_field]

        # Normalize dates
        date_fields = ['planned_start_date', 'planned_end_date']
        for field in date_fields:
            if field in transformed:
                transformed[field] = self._normalize_date(transformed[field])

        # Calculate duration
        if 'planned_start_date' in transformed and 'planned_end_date' in transformed:
            try:
                start = datetime.fromisoformat(transformed['planned_start_date'])
                end = datetime.fromisoformat(transformed['planned_end_date'])
                duration = (end - start).total_seconds() / 3600  # Hours
                transformed['planned_duration_hours'] = duration
            except (ValueError, TypeError):
                pass

        return transformed

    def _normalize_date(self, date_value: Any) -> str:
        """Normalize date to ISO format."""
        if isinstance(date_value, datetime):
            return date_value.isoformat()
        elif isinstance(date_value, str):
            try:
                # Try parsing various date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d']:
                    try:
                        dt = datetime.strptime(date_value, fmt)
                        return dt.isoformat()
                    except ValueError:
                        continue
                # If no format matches, return as-is
                return date_value
            except Exception:
                return date_value
        else:
            return str(date_value)


class AgentMessageFormatter:
    """
    Formats messages for agent communication.

    Ensures consistent message format across all agent interactions.
    """

    def __init__(self):
        """Initialize message formatter."""
        self.metrics = TransformationMetrics()

    def format_command_message(
        self,
        command: str,
        parameters: Dict[str, Any],
        target_agents: List[str],
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Format command message for agents.

        Args:
            command: Command to execute
            parameters: Command parameters
            target_agents: List of target agent IDs
            priority: Message priority

        Returns:
            Formatted command message
        """
        message = {
            'message_id': self._generate_message_id(),
            'message_type': 'command',
            'source_agent': 'GL-001',
            'target_agents': target_agents,
            'timestamp': DeterministicClock.utcnow().isoformat(),
            'priority': priority,
            'payload': {
                'command': command,
                'parameters': parameters
            },
            'requires_ack': True,
            'timeout_seconds': 30
        }

        return message

    def format_query_message(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        target_agents: List[str] = None
    ) -> Dict[str, Any]:
        """
        Format query message for agents.

        Args:
            query: Query string
            filters: Optional query filters
            target_agents: Optional target agents

        Returns:
            Formatted query message
        """
        message = {
            'message_id': self._generate_message_id(),
            'message_type': 'query',
            'source_agent': 'GL-001',
            'target_agents': target_agents or ['*'],
            'timestamp': DeterministicClock.utcnow().isoformat(),
            'priority': 'normal',
            'payload': {
                'query': query,
                'filters': filters or {}
            },
            'requires_response': True,
            'timeout_seconds': 10
        }

        return message

    def format_response_message(
        self,
        request_id: str,
        response_data: Any,
        status: str = "success"
    ) -> Dict[str, Any]:
        """
        Format response message.

        Args:
            request_id: Original request ID
            response_data: Response data
            status: Response status

        Returns:
            Formatted response message
        """
        message = {
            'message_id': self._generate_message_id(),
            'message_type': 'response',
            'source_agent': 'GL-001',
            'correlation_id': request_id,
            'timestamp': DeterministicClock.utcnow().isoformat(),
            'status': status,
            'payload': response_data
        }

        return message

    def format_event_message(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        severity: str = "info"
    ) -> Dict[str, Any]:
        """
        Format event notification message.

        Args:
            event_type: Type of event
            event_data: Event data
            severity: Event severity

        Returns:
            Formatted event message
        """
        message = {
            'message_id': self._generate_message_id(),
            'message_type': 'event',
            'source_agent': 'GL-001',
            'target_agents': ['*'],  # Broadcast
            'timestamp': DeterministicClock.utcnow().isoformat(),
            'severity': severity,
            'payload': {
                'event_type': event_type,
                'data': event_data
            },
            'requires_ack': severity in ['critical', 'error']
        }

        return message

    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        import uuid
        return f"GL001_{DeterministicClock.utcnow().strftime('%Y%m%d%H%M%S')}_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}"


# Example usage
def main():
    """Example data transformer usage."""

    # SCADA transformation
    scada_transformer = SCADADataTransformer()

    raw_sensor_data = {
        'sensor_id': 'TEMP_001',
        'value': 185.5,
        'unit': 'fahrenheit',
        'timestamp': '2024-01-15T10:30:00Z',
        'calibration_factor': 1.02,
        'min_value': 0,
        'max_value': 300
    }

    transformed = scada_transformer.transform_sensor_reading(raw_sensor_data)
    print(f"Transformed SCADA data: {json.dumps(transformed, default=str, indent=2)}")

    # Normalize to Celsius
    normalized = scada_transformer.normalize_units(transformed, 'celsius')
    print(f"Normalized temperature: {normalized['value']:.2f} {normalized['unit']}")

    # ERP transformation
    erp_transformer = ERPDataTransformer()

    raw_erp_data = {
        'PlantCode': 'PLANT01',
        'Date': '2024-01-15',
        'EnergyType': 'Electricity',
        'Consumption': 15000,
        'Unit': 'kWh',
        'Cost': 2250,
        'Currency': 'USD'
    }

    transformed_erp = erp_transformer.transform_energy_consumption(raw_erp_data)
    print(f"Transformed ERP data: {json.dumps(transformed_erp, default=str, indent=2)}")

    # Message formatting
    formatter = AgentMessageFormatter()

    command_msg = formatter.format_command_message(
        command='optimize_efficiency',
        parameters={'target': 0.95, 'mode': 'aggressive'},
        target_agents=['GL-002', 'GL-003'],
        priority='high'
    )

    print(f"Formatted command message: {json.dumps(command_msg, indent=2)}")


if __name__ == "__main__":
    main()