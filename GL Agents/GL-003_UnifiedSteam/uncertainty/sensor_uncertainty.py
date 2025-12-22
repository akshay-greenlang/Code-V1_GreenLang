"""
Sensor Uncertainty Management for GL-003 UNIFIEDSTEAM SteamSystemOptimizer.

This module manages sensor-level uncertainty metadata including calibration
tracking, drift modeling, and time-degraded uncertainty computation.

Zero-Hallucination Guarantee:
- All uncertainty calculations are deterministic formulas
- Drift models based on ISO 17025 and industrial standards
- Complete audit trail for all sensor registrations and updates
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import logging
import threading

from .uncertainty_models import (
    SensorUncertainty,
    SensorRegistration,
    SensorFlag,
    DriftClass,
    UncertainValue,
    DistributionType
)


logger = logging.getLogger(__name__)


# Default drift rates by class (percent per month)
DEFAULT_DRIFT_RATES: Dict[DriftClass, float] = {
    DriftClass.CLASS_A: 0.05,   # Precision instruments
    DriftClass.CLASS_B: 0.25,   # Standard industrial
    DriftClass.CLASS_C: 1.0,    # Harsh environment
    DriftClass.CLASS_D: 2.5,    # Requires frequent calibration
}

# Calibration interval recommendations by drift class (months)
RECOMMENDED_CALIBRATION_INTERVALS: Dict[DriftClass, int] = {
    DriftClass.CLASS_A: 24,     # 2 years
    DriftClass.CLASS_B: 12,     # 1 year
    DriftClass.CLASS_C: 6,      # 6 months
    DriftClass.CLASS_D: 3,      # 3 months
}

# Uncertainty thresholds for flagging (percent)
UNCERTAINTY_THRESHOLDS = {
    "warning": 3.0,      # Generate warning
    "high": 5.0,         # High uncertainty flag
    "critical": 10.0,    # Critical - require operator attention
    "unusable": 20.0,    # Sensor data unreliable
}


@dataclass
class CalibrationRecord:
    """Record of a sensor calibration event."""
    sensor_id: str
    calibration_date: datetime
    accuracy_achieved: float
    certificate_id: str
    calibrator_id: str
    method: str
    reference_standard: str
    environmental_conditions: Dict[str, float]
    notes: str = ""
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash."""
        if not self.provenance_hash:
            hash_data = {
                "sensor_id": self.sensor_id,
                "calibration_date": self.calibration_date.isoformat(),
                "accuracy_achieved": self.accuracy_achieved,
                "certificate_id": self.certificate_id,
                "calibrator_id": self.calibrator_id
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


class SensorUncertaintyManager:
    """
    Manages sensor uncertainty metadata and time-degraded uncertainty computation.

    This class maintains a registry of all sensors with their uncertainty
    profiles, tracks calibration history, and computes current uncertainty
    based on time since last calibration.

    Thread-safe for concurrent access.

    Example:
        manager = SensorUncertaintyManager()

        # Register a temperature sensor
        reg = manager.register_sensor(
            sensor_id="TT-101",
            accuracy_percent=0.5,
            calibration_date=datetime(2024, 1, 15),
            drift_class=DriftClass.CLASS_B,
            sensor_type="temperature",
            manufacturer="Rosemount"
        )

        # Get current uncertainty (includes drift)
        uncertainty = manager.get_sensor_uncertainty("TT-101")
        print(f"Current uncertainty: {uncertainty.current_uncertainty}%")

        # Check for high-uncertainty sensors
        flags = manager.flag_high_uncertainty_sensors()
    """

    def __init__(self):
        """Initialize the sensor uncertainty manager."""
        self._sensors: Dict[str, SensorRegistration] = {}
        self._calibration_history: Dict[str, List[CalibrationRecord]] = {}
        self._lock = threading.RLock()
        self._audit_log: List[Dict] = []

    def register_sensor(
        self,
        sensor_id: str,
        accuracy_percent: float,
        calibration_date: datetime,
        drift_class: DriftClass,
        sensor_type: str = "unknown",
        manufacturer: str = "",
        model: str = "",
        serial_number: str = "",
        installation_date: Optional[datetime] = None,
        location: str = "",
        measurement_range: Optional[Tuple[float, float]] = None,
        operating_temp_range: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> SensorRegistration:
        """
        Register a sensor with the uncertainty tracking system.

        Args:
            sensor_id: Unique sensor identifier (e.g., "TT-101", "PT-201")
            accuracy_percent: Manufacturer-specified accuracy (% of reading)
            calibration_date: Date of most recent calibration
            drift_class: Drift behavior classification
            sensor_type: Type of sensor (temperature, pressure, flow, etc.)
            manufacturer: Sensor manufacturer name
            model: Sensor model number
            serial_number: Unique serial number
            installation_date: When sensor was installed
            location: Physical location identifier
            measurement_range: Min/max measurement range
            operating_temp_range: Operating temperature range (C)
            metadata: Additional metadata

        Returns:
            SensorRegistration record

        Raises:
            ValueError: If sensor_id already registered or invalid parameters
        """
        with self._lock:
            if sensor_id in self._sensors:
                raise ValueError(f"Sensor {sensor_id} already registered")

            if accuracy_percent <= 0:
                raise ValueError(f"Accuracy must be positive: {accuracy_percent}")

            if calibration_date > datetime.utcnow():
                raise ValueError("Calibration date cannot be in the future")

            # Get drift rate for class
            drift_rate = DEFAULT_DRIFT_RATES[drift_class]

            # Compute current time-degraded uncertainty
            months_since_cal = (datetime.utcnow() - calibration_date).days / 30.44
            current_uncertainty = accuracy_percent + (drift_rate * months_since_cal)

            # Set defaults
            measurement_range = measurement_range or (0.0, float('inf'))
            operating_temp_range = operating_temp_range or (-40.0, 85.0)
            installation_date = installation_date or datetime.utcnow()
            metadata = metadata or {}

            # Create sensor uncertainty profile
            sensor_uncertainty = SensorUncertainty(
                sensor_id=sensor_id,
                base_accuracy=accuracy_percent,
                drift_rate=drift_rate,
                drift_class=drift_class,
                last_calibration=calibration_date,
                calibration_accuracy=accuracy_percent,
                current_uncertainty=current_uncertainty,
                measurement_range_min=measurement_range[0],
                measurement_range_max=measurement_range[1],
                operating_temperature_min=operating_temp_range[0],
                operating_temperature_max=operating_temp_range[1]
            )

            # Create registration record
            registration = SensorRegistration(
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                manufacturer=manufacturer,
                model=model,
                serial_number=serial_number,
                installation_date=installation_date,
                location=location,
                uncertainty=sensor_uncertainty,
                metadata=metadata
            )

            # Store registration
            self._sensors[sensor_id] = registration
            self._calibration_history[sensor_id] = []

            # Log audit event
            self._log_audit_event(
                event_type="sensor_registered",
                sensor_id=sensor_id,
                details={
                    "accuracy_percent": accuracy_percent,
                    "drift_class": drift_class.value,
                    "calibration_date": calibration_date.isoformat()
                }
            )

            logger.info(
                f"Registered sensor {sensor_id}: accuracy={accuracy_percent}%, "
                f"drift_class={drift_class.value}"
            )

            return registration

    def get_sensor_uncertainty(self, sensor_id: str) -> SensorUncertainty:
        """
        Get current uncertainty profile for a sensor.

        The returned uncertainty includes time degradation since
        last calibration based on the sensor's drift class.

        Args:
            sensor_id: Sensor identifier

        Returns:
            SensorUncertainty with current (degraded) uncertainty

        Raises:
            KeyError: If sensor not found
        """
        with self._lock:
            if sensor_id not in self._sensors:
                raise KeyError(f"Sensor {sensor_id} not registered")

            registration = self._sensors[sensor_id]
            base_uncertainty = registration.uncertainty

            # Compute current time-degraded uncertainty
            current_uncertainty = self.compute_time_degraded_uncertainty(
                sensor_id,
                datetime.utcnow()
            )

            # Return updated uncertainty object
            return SensorUncertainty(
                sensor_id=base_uncertainty.sensor_id,
                base_accuracy=base_uncertainty.base_accuracy,
                drift_rate=base_uncertainty.drift_rate,
                drift_class=base_uncertainty.drift_class,
                last_calibration=base_uncertainty.last_calibration,
                calibration_accuracy=base_uncertainty.calibration_accuracy,
                current_uncertainty=current_uncertainty,
                measurement_range_min=base_uncertainty.measurement_range_min,
                measurement_range_max=base_uncertainty.measurement_range_max,
                operating_temperature_min=base_uncertainty.operating_temperature_min,
                operating_temperature_max=base_uncertainty.operating_temperature_max,
                calibration_certificate_id=base_uncertainty.calibration_certificate_id
            )

    def update_calibration(
        self,
        sensor_id: str,
        new_calibration_date: datetime,
        new_accuracy: float,
        certificate_id: str = "",
        calibrator_id: str = "",
        method: str = "",
        reference_standard: str = "",
        environmental_conditions: Optional[Dict[str, float]] = None,
        notes: str = ""
    ) -> None:
        """
        Update sensor calibration information.

        Records calibration history and resets the time-degraded
        uncertainty to the new calibration accuracy.

        Args:
            sensor_id: Sensor identifier
            new_calibration_date: Date of new calibration
            new_accuracy: Accuracy achieved at calibration (%)
            certificate_id: Calibration certificate identifier
            calibrator_id: ID of person/company performing calibration
            method: Calibration method used
            reference_standard: Reference standard used
            environmental_conditions: Temperature, humidity, etc.
            notes: Additional notes

        Raises:
            KeyError: If sensor not found
            ValueError: If invalid parameters
        """
        with self._lock:
            if sensor_id not in self._sensors:
                raise KeyError(f"Sensor {sensor_id} not registered")

            if new_accuracy <= 0:
                raise ValueError(f"Accuracy must be positive: {new_accuracy}")

            if new_calibration_date > datetime.utcnow():
                raise ValueError("Calibration date cannot be in the future")

            old_registration = self._sensors[sensor_id]
            old_uncertainty = old_registration.uncertainty

            if new_calibration_date < old_uncertainty.last_calibration:
                raise ValueError(
                    f"New calibration date {new_calibration_date} is before "
                    f"previous calibration {old_uncertainty.last_calibration}"
                )

            # Create calibration record
            cal_record = CalibrationRecord(
                sensor_id=sensor_id,
                calibration_date=new_calibration_date,
                accuracy_achieved=new_accuracy,
                certificate_id=certificate_id,
                calibrator_id=calibrator_id,
                method=method,
                reference_standard=reference_standard,
                environmental_conditions=environmental_conditions or {},
                notes=notes
            )

            # Store calibration history
            self._calibration_history[sensor_id].append(cal_record)

            # Create updated uncertainty profile
            new_sensor_uncertainty = SensorUncertainty(
                sensor_id=old_uncertainty.sensor_id,
                base_accuracy=old_uncertainty.base_accuracy,
                drift_rate=old_uncertainty.drift_rate,
                drift_class=old_uncertainty.drift_class,
                last_calibration=new_calibration_date,
                calibration_accuracy=new_accuracy,
                current_uncertainty=new_accuracy,  # Reset to calibration accuracy
                measurement_range_min=old_uncertainty.measurement_range_min,
                measurement_range_max=old_uncertainty.measurement_range_max,
                operating_temperature_min=old_uncertainty.operating_temperature_min,
                operating_temperature_max=old_uncertainty.operating_temperature_max,
                calibration_certificate_id=certificate_id
            )

            # Update registration
            new_registration = SensorRegistration(
                sensor_id=old_registration.sensor_id,
                sensor_type=old_registration.sensor_type,
                manufacturer=old_registration.manufacturer,
                model=old_registration.model,
                serial_number=old_registration.serial_number,
                installation_date=old_registration.installation_date,
                location=old_registration.location,
                uncertainty=new_sensor_uncertainty,
                registration_timestamp=old_registration.registration_timestamp,
                metadata=old_registration.metadata
            )

            self._sensors[sensor_id] = new_registration

            # Log audit event
            self._log_audit_event(
                event_type="calibration_updated",
                sensor_id=sensor_id,
                details={
                    "new_calibration_date": new_calibration_date.isoformat(),
                    "new_accuracy": new_accuracy,
                    "old_accuracy": old_uncertainty.calibration_accuracy,
                    "certificate_id": certificate_id
                }
            )

            logger.info(
                f"Updated calibration for sensor {sensor_id}: "
                f"accuracy={new_accuracy}%, date={new_calibration_date}"
            )

    def compute_time_degraded_uncertainty(
        self,
        sensor_id: str,
        current_time: datetime
    ) -> float:
        """
        Compute uncertainty including degradation since last calibration.

        Uses linear drift model: U(t) = U_cal + drift_rate * t
        where t is time since calibration in months.

        Args:
            sensor_id: Sensor identifier
            current_time: Time at which to compute uncertainty

        Returns:
            Time-degraded uncertainty (%)

        Raises:
            KeyError: If sensor not found
        """
        with self._lock:
            if sensor_id not in self._sensors:
                raise KeyError(f"Sensor {sensor_id} not registered")

            registration = self._sensors[sensor_id]
            uncertainty = registration.uncertainty

            # Calculate months since calibration
            delta = current_time - uncertainty.last_calibration
            months_since_calibration = delta.days / 30.44

            # Linear drift model (deterministic)
            degraded_uncertainty = (
                uncertainty.calibration_accuracy +
                uncertainty.drift_rate * months_since_calibration
            )

            # Cap at a reasonable maximum (equipment should be recalibrated)
            max_uncertainty = uncertainty.base_accuracy * 10.0
            degraded_uncertainty = min(degraded_uncertainty, max_uncertainty)

            return round(degraded_uncertainty, 4)

    def flag_high_uncertainty_sensors(
        self,
        warning_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None
    ) -> List[SensorFlag]:
        """
        Identify sensors with uncertainty exceeding thresholds.

        Scans all registered sensors and returns flags for those
        requiring attention based on configurable thresholds.

        Args:
            warning_threshold: Warning level threshold (%)
            high_threshold: High uncertainty threshold (%)
            critical_threshold: Critical uncertainty threshold (%)

        Returns:
            List of SensorFlag objects for sensors exceeding thresholds
        """
        warning_threshold = warning_threshold or UNCERTAINTY_THRESHOLDS["warning"]
        high_threshold = high_threshold or UNCERTAINTY_THRESHOLDS["high"]
        critical_threshold = critical_threshold or UNCERTAINTY_THRESHOLDS["critical"]

        flags = []
        current_time = datetime.utcnow()

        with self._lock:
            for sensor_id, registration in self._sensors.items():
                # Compute current uncertainty
                current_uncertainty = self.compute_time_degraded_uncertainty(
                    sensor_id,
                    current_time
                )

                uncertainty_profile = registration.uncertainty
                days_since_cal = (
                    current_time - uncertainty_profile.last_calibration
                ).days

                # Determine flag priority and action
                if current_uncertainty >= critical_threshold:
                    priority = 1
                    threshold_exceeded = "critical"
                    recommended_action = (
                        f"URGENT: Sensor {sensor_id} uncertainty is critical "
                        f"({current_uncertainty:.2f}%). Immediate recalibration required."
                    )
                elif current_uncertainty >= high_threshold:
                    priority = 2
                    threshold_exceeded = "high"
                    recommended_action = (
                        f"Sensor {sensor_id} uncertainty is high "
                        f"({current_uncertainty:.2f}%). Schedule recalibration."
                    )
                elif current_uncertainty >= warning_threshold:
                    priority = 3
                    threshold_exceeded = "warning"
                    recommended_action = (
                        f"Sensor {sensor_id} uncertainty approaching limit "
                        f"({current_uncertainty:.2f}%). Monitor closely."
                    )
                else:
                    # No flag needed
                    continue

                # Check calibration due
                recommended_interval = RECOMMENDED_CALIBRATION_INTERVALS.get(
                    uncertainty_profile.drift_class,
                    12
                )
                if days_since_cal > recommended_interval * 30:
                    recommended_action += (
                        f" Calibration overdue by "
                        f"{days_since_cal - recommended_interval * 30} days."
                    )

                flag = SensorFlag(
                    sensor_id=sensor_id,
                    current_uncertainty=current_uncertainty,
                    threshold_exceeded=threshold_exceeded,
                    days_since_calibration=days_since_cal,
                    recommended_action=recommended_action,
                    priority=priority
                )

                flags.append(flag)

        # Sort by priority (critical first)
        flags.sort(key=lambda f: (f.priority, -f.current_uncertainty))

        return flags

    def get_measurement_uncertainty(
        self,
        sensor_id: str,
        measured_value: float,
        current_time: Optional[datetime] = None
    ) -> UncertainValue:
        """
        Create UncertainValue for a measurement from a registered sensor.

        Combines the measured value with the sensor's current uncertainty
        to produce a complete uncertain value ready for propagation.

        Args:
            sensor_id: Sensor identifier
            measured_value: The raw measured value
            current_time: Time of measurement (default: now)

        Returns:
            UncertainValue with proper uncertainty bounds

        Raises:
            KeyError: If sensor not found
        """
        current_time = current_time or datetime.utcnow()

        with self._lock:
            if sensor_id not in self._sensors:
                raise KeyError(f"Sensor {sensor_id} not registered")

            registration = self._sensors[sensor_id]
            uncertainty_percent = self.compute_time_degraded_uncertainty(
                sensor_id,
                current_time
            )

            return UncertainValue.from_measurement(
                value=measured_value,
                uncertainty_percent=uncertainty_percent,
                unit=registration.metadata.get("unit", ""),
                source_id=sensor_id,
                distribution=DistributionType.NORMAL
            )

    def get_sensors_due_for_calibration(
        self,
        days_warning: int = 30
    ) -> List[Tuple[str, int, datetime]]:
        """
        Get list of sensors approaching or past calibration due date.

        Args:
            days_warning: Days before due date to include in warning

        Returns:
            List of (sensor_id, days_until_due, due_date) tuples
        """
        due_sensors = []
        current_time = datetime.utcnow()

        with self._lock:
            for sensor_id, registration in self._sensors.items():
                uncertainty = registration.uncertainty

                # Get recommended interval for drift class
                interval_months = RECOMMENDED_CALIBRATION_INTERVALS.get(
                    uncertainty.drift_class,
                    12
                )

                due_date = uncertainty.last_calibration + timedelta(
                    days=interval_months * 30
                )
                days_until_due = (due_date - current_time).days

                if days_until_due <= days_warning:
                    due_sensors.append((sensor_id, days_until_due, due_date))

        # Sort by days until due (most urgent first)
        due_sensors.sort(key=lambda x: x[1])

        return due_sensors

    def get_calibration_history(
        self,
        sensor_id: str
    ) -> List[CalibrationRecord]:
        """
        Get calibration history for a sensor.

        Args:
            sensor_id: Sensor identifier

        Returns:
            List of calibration records, newest first

        Raises:
            KeyError: If sensor not found
        """
        with self._lock:
            if sensor_id not in self._sensors:
                raise KeyError(f"Sensor {sensor_id} not registered")

            history = self._calibration_history.get(sensor_id, [])
            return sorted(history, key=lambda r: r.calibration_date, reverse=True)

    def get_all_sensors(self) -> List[SensorRegistration]:
        """
        Get all registered sensors.

        Returns:
            List of SensorRegistration objects
        """
        with self._lock:
            return list(self._sensors.values())

    def get_sensor_count_by_drift_class(self) -> Dict[DriftClass, int]:
        """
        Get count of sensors by drift class.

        Returns:
            Dictionary mapping drift class to count
        """
        counts = {dc: 0 for dc in DriftClass}

        with self._lock:
            for registration in self._sensors.values():
                drift_class = registration.uncertainty.drift_class
                counts[drift_class] += 1

        return counts

    def remove_sensor(self, sensor_id: str) -> None:
        """
        Remove a sensor from the registry.

        Args:
            sensor_id: Sensor identifier

        Raises:
            KeyError: If sensor not found
        """
        with self._lock:
            if sensor_id not in self._sensors:
                raise KeyError(f"Sensor {sensor_id} not registered")

            del self._sensors[sensor_id]

            if sensor_id in self._calibration_history:
                del self._calibration_history[sensor_id]

            self._log_audit_event(
                event_type="sensor_removed",
                sensor_id=sensor_id,
                details={}
            )

            logger.info(f"Removed sensor {sensor_id}")

    def _log_audit_event(
        self,
        event_type: str,
        sensor_id: str,
        details: Dict
    ) -> None:
        """Log an audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "sensor_id": sensor_id,
            "details": details
        }

        # Compute event hash
        event_str = json.dumps(event, sort_keys=True)
        event["hash"] = hashlib.sha256(event_str.encode()).hexdigest()

        self._audit_log.append(event)

    def get_audit_log(
        self,
        sensor_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get audit log entries with optional filtering.

        Args:
            sensor_id: Filter by sensor
            event_type: Filter by event type
            since: Filter events after this time

        Returns:
            List of audit log entries
        """
        with self._lock:
            entries = self._audit_log.copy()

        if sensor_id:
            entries = [e for e in entries if e["sensor_id"] == sensor_id]

        if event_type:
            entries = [e for e in entries if e["event_type"] == event_type]

        if since:
            since_str = since.isoformat()
            entries = [e for e in entries if e["timestamp"] >= since_str]

        return entries

    def export_sensor_registry(self) -> Dict:
        """
        Export complete sensor registry for backup/audit.

        Returns:
            Dictionary with all sensor data and metadata
        """
        with self._lock:
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "sensor_count": len(self._sensors),
                "sensors": {},
                "calibration_history": {}
            }

            for sensor_id, registration in self._sensors.items():
                export_data["sensors"][sensor_id] = {
                    "sensor_id": registration.sensor_id,
                    "sensor_type": registration.sensor_type,
                    "manufacturer": registration.manufacturer,
                    "model": registration.model,
                    "serial_number": registration.serial_number,
                    "installation_date": registration.installation_date.isoformat(),
                    "location": registration.location,
                    "uncertainty": {
                        "base_accuracy": registration.uncertainty.base_accuracy,
                        "drift_rate": registration.uncertainty.drift_rate,
                        "drift_class": registration.uncertainty.drift_class.value,
                        "last_calibration": registration.uncertainty.last_calibration.isoformat(),
                        "calibration_accuracy": registration.uncertainty.calibration_accuracy,
                        "current_uncertainty": self.compute_time_degraded_uncertainty(
                            sensor_id, datetime.utcnow()
                        )
                    },
                    "metadata": registration.metadata
                }

            for sensor_id, history in self._calibration_history.items():
                export_data["calibration_history"][sensor_id] = [
                    {
                        "calibration_date": record.calibration_date.isoformat(),
                        "accuracy_achieved": record.accuracy_achieved,
                        "certificate_id": record.certificate_id,
                        "calibrator_id": record.calibrator_id,
                        "method": record.method,
                        "reference_standard": record.reference_standard,
                        "provenance_hash": record.provenance_hash
                    }
                    for record in history
                ]

            # Compute export hash
            export_str = json.dumps(export_data, sort_keys=True, default=str)
            export_data["export_hash"] = hashlib.sha256(export_str.encode()).hexdigest()

            return export_data
