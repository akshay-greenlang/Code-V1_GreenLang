"""
GL-013 PREDICTMAINT - Example Usage
Demonstrates all capabilities of the predictive maintenance agent.

This module provides comprehensive examples showing how to use the GL-013
PREDICTMAINT agent for equipment diagnostics, failure prediction, maintenance
scheduling, and health monitoring.

Examples cover:
    - Equipment diagnostics with vibration and thermal analysis
    - Failure probability prediction and RUL estimation
    - Maintenance schedule optimization
    - Batch processing for multiple equipment
    - Anomaly detection
    - Spare parts forecasting
    - Full workflow integration

Usage:
    Run directly to execute all examples:
        python example_usage.py

    Or import specific examples:
        from example_usage import example_diagnose_equipment
        asyncio.run(example_diagnose_equipment())

Author: GreenLang API Team
Version: 1.0.0
Agent ID: GL-013
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# SIMULATED API CLIENT
# =============================================================================

class PredictMaintClient:
    """
    Simulated client for GL-013 PREDICTMAINT API.

    In production, this would use httpx or aiohttp to call the REST API.
    For demonstration, it directly imports and uses the API functions.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.

        Args:
            base_url: Base URL for the API
        """
        self.base_url = base_url
        self.api_key = "demo-api-key"

        # For demo purposes, import calculators directly
        from .calculators import (
            RULCalculator,
            FailureProbabilityCalculator,
            VibrationAnalyzer,
            ThermalDegradationCalculator,
            MaintenanceScheduler,
            SparePartsCalculator,
            AnomalyDetector,
        )
        from .tools import PredictiveMaintenanceTools

        self.rul_calculator = RULCalculator()
        self.failure_calculator = FailureProbabilityCalculator()
        self.vibration_analyzer = VibrationAnalyzer()
        self.thermal_calculator = ThermalDegradationCalculator()
        self.maintenance_scheduler = MaintenanceScheduler()
        self.spare_parts_calculator = SparePartsCalculator()
        self.anomaly_detector = AnomalyDetector()
        self.tools = PredictiveMaintenanceTools()

    async def diagnose_equipment(
        self,
        equipment_id: str,
        equipment_type: str,
        vibration_velocity_mm_s: Optional[float] = None,
        temperature_c: Optional[float] = None,
        machine_class: str = "class_ii"
    ) -> Dict[str, Any]:
        """
        Diagnose equipment condition.

        Args:
            equipment_id: Equipment identifier
            equipment_type: Type of equipment (pump, motor, etc.)
            vibration_velocity_mm_s: Vibration velocity in mm/s
            temperature_c: Operating temperature in Celsius
            machine_class: ISO 10816 machine class

        Returns:
            Diagnosis results dictionary
        """
        results = {
            "equipment_id": equipment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health_score": 100.0,
            "health_status": "healthy",
            "fault_indicators": [],
            "recommendations": []
        }

        # Vibration analysis
        if vibration_velocity_mm_s is not None:
            vib_result = self.vibration_analyzer.assess_severity(
                velocity_rms=Decimal(str(vibration_velocity_mm_s)),
                machine_class=machine_class
            )
            results["vibration"] = {
                "velocity_mm_s": vibration_velocity_mm_s,
                "zone": vib_result.zone.name,
                "severity": vib_result.severity
            }

            # Adjust health score
            if vib_result.zone.name == "B":
                results["health_score"] -= 10
            elif vib_result.zone.name == "C":
                results["health_score"] -= 30
                results["fault_indicators"].append({
                    "type": "excessive_vibration",
                    "severity": "high"
                })
            elif vib_result.zone.name == "D":
                results["health_score"] -= 50
                results["fault_indicators"].append({
                    "type": "critical_vibration",
                    "severity": "critical"
                })

        # Thermal analysis
        if temperature_c is not None:
            thermal_result = self.thermal_calculator.calculate_aging_acceleration(
                operating_temp_c=Decimal(str(temperature_c)),
                reference_temp_c=Decimal("105")
            )
            results["thermal"] = {
                "temperature_c": temperature_c,
                "aging_factor": float(thermal_result.acceleration_factor)
            }

            # Adjust health score
            aging = float(thermal_result.acceleration_factor)
            if aging > 3.0:
                results["health_score"] -= 30
                results["fault_indicators"].append({
                    "type": "thermal_stress",
                    "severity": "high"
                })
            elif aging > 2.0:
                results["health_score"] -= 20
            elif aging > 1.5:
                results["health_score"] -= 10

        # Classify health status
        score = results["health_score"]
        if score >= 90:
            results["health_status"] = "healthy"
        elif score >= 70:
            results["health_status"] = "monitored"
        elif score >= 50:
            results["health_status"] = "degraded"
        elif score >= 30:
            results["health_status"] = "at_risk"
        else:
            results["health_status"] = "critical"

        # Generate recommendations
        if results["health_status"] == "healthy":
            results["recommendations"].append("Continue normal operation")
        elif results["health_status"] == "monitored":
            results["recommendations"].append("Increase monitoring frequency")
        elif results["health_status"] == "degraded":
            results["recommendations"].append("Plan maintenance within 30 days")
        elif results["health_status"] == "at_risk":
            results["recommendations"].append("Schedule maintenance within 7 days")
        else:
            results["recommendations"].append("Immediate maintenance required")

        return results

    async def predict_failure(
        self,
        equipment_id: str,
        equipment_type: str,
        operating_hours: float,
        prediction_horizon_hours: float = 8760.0
    ) -> Dict[str, Any]:
        """
        Predict equipment failure probability and RUL.

        Args:
            equipment_id: Equipment identifier
            equipment_type: Type of equipment
            operating_hours: Current operating hours
            prediction_horizon_hours: Prediction horizon

        Returns:
            Prediction results dictionary
        """
        # Calculate failure probability
        prob_result = self.failure_calculator.calculate_weibull_probability(
            time_hours=Decimal(str(operating_hours + prediction_horizon_hours)),
            equipment_type=equipment_type,
            current_age_hours=Decimal(str(operating_hours))
        )

        # Calculate RUL
        rul_result = self.rul_calculator.calculate_weibull_rul(
            equipment_type=equipment_type,
            operating_hours=Decimal(str(operating_hours)),
            target_reliability="0.5"
        )

        # Classify risk
        prob = float(prob_result.probability)
        if prob < 0.05:
            risk_level = "low"
        elif prob < 0.20:
            risk_level = "medium"
        elif prob < 0.50:
            risk_level = "high"
        else:
            risk_level = "critical"

        return {
            "equipment_id": equipment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "failure_probability": prob,
            "rul_hours": float(rul_result.rul_hours),
            "rul_days": float(rul_result.rul_hours) / 24.0,
            "risk_level": risk_level,
            "hazard_rate": float(prob_result.hazard_rate),
            "distribution": "weibull"
        }

    async def schedule_maintenance(
        self,
        equipment_id: str,
        equipment_type: str,
        current_health_score: float,
        operating_hours: float,
        preventive_cost: float = 1000.0,
        failure_cost: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Schedule optimal maintenance.

        Args:
            equipment_id: Equipment identifier
            equipment_type: Type of equipment
            current_health_score: Current health score (0-100)
            operating_hours: Current operating hours
            preventive_cost: Cost of preventive maintenance
            failure_cost: Cost of failure/corrective maintenance

        Returns:
            Schedule results dictionary
        """
        # Calculate optimal interval
        interval_result = self.maintenance_scheduler.calculate_optimal_interval(
            equipment_type=equipment_type,
            preventive_cost=Decimal(str(preventive_cost)),
            failure_cost=Decimal(str(failure_cost))
        )

        optimal_interval = float(interval_result.optimal_interval_hours)
        next_hours = optimal_interval - (operating_hours % optimal_interval)
        recommended_date = datetime.now(timezone.utc) + timedelta(hours=next_hours)

        # Determine priority
        if current_health_score < 30:
            priority = "emergency"
        elif current_health_score < 50:
            priority = "urgent"
        elif current_health_score < 70:
            priority = "high"
        elif current_health_score < 85:
            priority = "medium"
        else:
            priority = "low"

        return {
            "equipment_id": equipment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommended_date": recommended_date.isoformat(),
            "optimal_interval_hours": optimal_interval,
            "priority": priority,
            "expected_cost": preventive_cost,
            "expected_savings": failure_cost - preventive_cost,
            "recommended_actions": [
                "Inspect bearings",
                "Check lubrication",
                "Verify alignment",
                "Test operational parameters"
            ]
        }

    async def detect_anomalies(
        self,
        equipment_id: str,
        sensor_readings: Dict[str, List[float]],
        sensitivity: float = 3.0
    ) -> Dict[str, Any]:
        """
        Detect anomalies in sensor data.

        Args:
            equipment_id: Equipment identifier
            sensor_readings: Dictionary of sensor readings by parameter
            sensitivity: Detection sensitivity (sigma multiplier)

        Returns:
            Anomaly detection results
        """
        contributing_factors = []
        max_score = 0.0
        anomaly_detected = False

        for param, readings in sensor_readings.items():
            if len(readings) < 3:
                continue

            # Calculate statistics
            mean_val = sum(readings) / len(readings)
            variance = sum((x - mean_val) ** 2 for x in readings) / len(readings)
            std_dev = variance ** 0.5 if variance > 0 else 0.001

            # Check latest
            latest = readings[-1]
            z_score = abs(latest - mean_val) / std_dev

            if z_score > sensitivity:
                anomaly_detected = True
                contributing_factors.append({
                    "parameter": param,
                    "z_score": round(z_score, 2),
                    "value": latest,
                    "mean": round(mean_val, 2)
                })

            max_score = max(max_score, min(1.0, z_score / (sensitivity * 2)))

        # Determine severity
        if max_score >= 0.8:
            severity = "critical"
        elif max_score >= 0.6:
            severity = "high"
        elif max_score >= 0.4:
            severity = "warning"
        else:
            severity = "info"

        return {
            "equipment_id": equipment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "anomaly_detected": anomaly_detected,
            "anomaly_score": round(max_score, 4),
            "severity": severity,
            "contributing_factors": contributing_factors
        }


# =============================================================================
# EXAMPLE: DIAGNOSE EQUIPMENT
# =============================================================================

async def example_diagnose_equipment():
    """
    Example: Diagnose equipment with vibration and thermal analysis.

    Demonstrates how to submit sensor data and receive a comprehensive
    equipment health diagnosis.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Equipment Diagnostics")
    print("=" * 70)

    client = PredictMaintClient()

    # Example 1: Healthy equipment
    print("\n--- Case 1: Healthy Equipment ---")
    result = await client.diagnose_equipment(
        equipment_id="PUMP-001",
        equipment_type="pump",
        vibration_velocity_mm_s=2.5,
        temperature_c=55.0,
        machine_class="class_ii"
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Health Score: {result['health_score']:.1f}")
    print(f"Health Status: {result['health_status']}")
    print(f"Vibration Zone: {result.get('vibration', {}).get('zone', 'N/A')}")
    print(f"Recommendations: {result['recommendations']}")

    # Example 2: Degraded equipment
    print("\n--- Case 2: Degraded Equipment ---")
    result = await client.diagnose_equipment(
        equipment_id="PUMP-002",
        equipment_type="pump",
        vibration_velocity_mm_s=5.5,
        temperature_c=75.0,
        machine_class="class_ii"
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Health Score: {result['health_score']:.1f}")
    print(f"Health Status: {result['health_status']}")
    print(f"Vibration Zone: {result.get('vibration', {}).get('zone', 'N/A')}")
    print(f"Fault Indicators: {len(result['fault_indicators'])} detected")
    print(f"Recommendations: {result['recommendations']}")

    # Example 3: Critical equipment
    print("\n--- Case 3: Critical Equipment ---")
    result = await client.diagnose_equipment(
        equipment_id="PUMP-003",
        equipment_type="pump",
        vibration_velocity_mm_s=12.0,
        temperature_c=110.0,
        machine_class="class_ii"
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Health Score: {result['health_score']:.1f}")
    print(f"Health Status: {result['health_status']}")
    print(f"Fault Indicators: {result['fault_indicators']}")
    print(f"Recommendations: {result['recommendations']}")

    return result


# =============================================================================
# EXAMPLE: PREDICT FAILURE
# =============================================================================

async def example_predict_failure():
    """
    Example: Predict equipment failure probability and RUL.

    Demonstrates failure prediction using Weibull reliability analysis.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Failure Prediction")
    print("=" * 70)

    client = PredictMaintClient()

    # Example 1: New equipment (low hours)
    print("\n--- Case 1: New Equipment (5,000 hours) ---")
    result = await client.predict_failure(
        equipment_id="MOTOR-001",
        equipment_type="motor",
        operating_hours=5000.0,
        prediction_horizon_hours=8760.0  # 1 year
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Operating Hours: 5,000")
    print(f"Prediction Horizon: 1 year")
    print(f"Failure Probability: {result['failure_probability']:.2%}")
    print(f"RUL: {result['rul_hours']:.0f} hours ({result['rul_days']:.0f} days)")
    print(f"Risk Level: {result['risk_level']}")

    # Example 2: Mid-life equipment
    print("\n--- Case 2: Mid-Life Equipment (25,000 hours) ---")
    result = await client.predict_failure(
        equipment_id="MOTOR-002",
        equipment_type="motor",
        operating_hours=25000.0,
        prediction_horizon_hours=8760.0
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Operating Hours: 25,000")
    print(f"Failure Probability: {result['failure_probability']:.2%}")
    print(f"RUL: {result['rul_hours']:.0f} hours ({result['rul_days']:.0f} days)")
    print(f"Risk Level: {result['risk_level']}")

    # Example 3: Aged equipment
    print("\n--- Case 3: Aged Equipment (45,000 hours) ---")
    result = await client.predict_failure(
        equipment_id="MOTOR-003",
        equipment_type="motor",
        operating_hours=45000.0,
        prediction_horizon_hours=8760.0
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Operating Hours: 45,000")
    print(f"Failure Probability: {result['failure_probability']:.2%}")
    print(f"RUL: {result['rul_hours']:.0f} hours ({result['rul_days']:.0f} days)")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Hazard Rate: {result['hazard_rate']:.6f} per hour")

    return result


# =============================================================================
# EXAMPLE: SCHEDULE MAINTENANCE
# =============================================================================

async def example_schedule_maintenance():
    """
    Example: Schedule optimal maintenance.

    Demonstrates cost-optimized maintenance scheduling.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Maintenance Scheduling")
    print("=" * 70)

    client = PredictMaintClient()

    # Example 1: Healthy equipment - routine maintenance
    print("\n--- Case 1: Routine Maintenance (Health Score: 85) ---")
    result = await client.schedule_maintenance(
        equipment_id="GEARBOX-001",
        equipment_type="gearbox",
        current_health_score=85.0,
        operating_hours=10000.0,
        preventive_cost=2500.0,
        failure_cost=75000.0
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Priority: {result['priority']}")
    print(f"Recommended Date: {result['recommended_date'][:10]}")
    print(f"Optimal Interval: {result['optimal_interval_hours']:.0f} hours")
    print(f"Expected Cost: ${result['expected_cost']:,.2f}")
    print(f"Expected Savings: ${result['expected_savings']:,.2f}")

    # Example 2: Degraded equipment - urgent maintenance
    print("\n--- Case 2: Urgent Maintenance (Health Score: 55) ---")
    result = await client.schedule_maintenance(
        equipment_id="GEARBOX-002",
        equipment_type="gearbox",
        current_health_score=55.0,
        operating_hours=35000.0,
        preventive_cost=2500.0,
        failure_cost=75000.0
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Priority: {result['priority']}")
    print(f"Recommended Date: {result['recommended_date'][:10]}")
    print(f"Recommended Actions:")
    for action in result['recommended_actions']:
        print(f"  - {action}")

    # Example 3: Critical equipment - emergency
    print("\n--- Case 3: Emergency Maintenance (Health Score: 25) ---")
    result = await client.schedule_maintenance(
        equipment_id="GEARBOX-003",
        equipment_type="gearbox",
        current_health_score=25.0,
        operating_hours=50000.0,
        preventive_cost=2500.0,
        failure_cost=75000.0
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Priority: {result['priority']}")
    print(f"Recommended Date: {result['recommended_date'][:10]}")
    print(f"Cost/Benefit Ratio: {result['expected_savings'] / result['expected_cost']:.1f}x")

    return result


# =============================================================================
# EXAMPLE: FULL WORKFLOW
# =============================================================================

async def example_full_workflow():
    """
    Example: Complete predictive maintenance workflow.

    Demonstrates the full cycle from data collection through action.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Full Predictive Maintenance Workflow")
    print("=" * 70)

    client = PredictMaintClient()
    equipment_id = "COMPRESSOR-001"

    # Step 1: Collect and analyze sensor data
    print("\n[Step 1] Collecting sensor data...")
    print("  Vibration: 4.2 mm/s RMS")
    print("  Temperature: 68 C")
    print("  Pressure: 8.5 bar")

    # Step 2: Diagnose current condition
    print("\n[Step 2] Diagnosing equipment condition...")
    diagnosis = await client.diagnose_equipment(
        equipment_id=equipment_id,
        equipment_type="compressor",
        vibration_velocity_mm_s=4.2,
        temperature_c=68.0,
        machine_class="class_ii"
    )
    print(f"  Health Score: {diagnosis['health_score']:.1f}")
    print(f"  Health Status: {diagnosis['health_status']}")
    print(f"  Vibration Zone: {diagnosis.get('vibration', {}).get('zone', 'N/A')}")

    # Step 3: Predict failure probability
    print("\n[Step 3] Predicting failure probability...")
    prediction = await client.predict_failure(
        equipment_id=equipment_id,
        equipment_type="compressor",
        operating_hours=28000.0,
        prediction_horizon_hours=2160.0  # 90 days
    )
    print(f"  90-Day Failure Probability: {prediction['failure_probability']:.2%}")
    print(f"  Remaining Useful Life: {prediction['rul_days']:.0f} days")
    print(f"  Risk Level: {prediction['risk_level']}")

    # Step 4: Schedule maintenance
    print("\n[Step 4] Scheduling maintenance...")
    schedule = await client.schedule_maintenance(
        equipment_id=equipment_id,
        equipment_type="compressor",
        current_health_score=diagnosis['health_score'],
        operating_hours=28000.0,
        preventive_cost=3500.0,
        failure_cost=150000.0
    )
    print(f"  Priority: {schedule['priority']}")
    print(f"  Recommended Date: {schedule['recommended_date'][:10]}")
    print(f"  Expected Savings: ${schedule['expected_savings']:,.2f}")

    # Step 5: Generate work order summary
    print("\n[Step 5] Work Order Summary")
    print("-" * 40)
    print(f"  Equipment ID: {equipment_id}")
    print(f"  Work Order Priority: {schedule['priority'].upper()}")
    print(f"  Scheduled Date: {schedule['recommended_date'][:10]}")
    print(f"  Estimated Duration: 4 hours")
    print(f"  Recommended Actions:")
    for action in schedule['recommended_actions']:
        print(f"    - {action}")

    print("\n[Workflow Complete]")
    print(f"  Total analysis time: <1 second")
    print(f"  Potential cost savings: ${schedule['expected_savings']:,.2f}")

    return {
        "diagnosis": diagnosis,
        "prediction": prediction,
        "schedule": schedule
    }


# =============================================================================
# EXAMPLE: BATCH PROCESSING
# =============================================================================

async def example_batch_processing():
    """
    Example: Process multiple equipment items in batch.

    Demonstrates efficient batch processing for fleet management.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Batch Processing")
    print("=" * 70)

    client = PredictMaintClient()

    # Define fleet
    equipment_fleet = [
        {"id": "FAN-001", "type": "fan", "hours": 15000, "vibration": 1.8, "temp": 45},
        {"id": "FAN-002", "type": "fan", "hours": 28000, "vibration": 3.5, "temp": 52},
        {"id": "FAN-003", "type": "fan", "hours": 42000, "vibration": 6.2, "temp": 65},
        {"id": "PUMP-101", "type": "pump", "hours": 8000, "vibration": 2.1, "temp": 48},
        {"id": "PUMP-102", "type": "pump", "hours": 35000, "vibration": 4.8, "temp": 58},
    ]

    print(f"\nProcessing {len(equipment_fleet)} equipment items...")

    results = []
    critical_count = 0
    at_risk_count = 0

    for equip in equipment_fleet:
        # Diagnose each equipment
        diagnosis = await client.diagnose_equipment(
            equipment_id=equip["id"],
            equipment_type=equip["type"],
            vibration_velocity_mm_s=equip["vibration"],
            temperature_c=equip["temp"]
        )

        # Predict failure
        prediction = await client.predict_failure(
            equipment_id=equip["id"],
            equipment_type=equip["type"],
            operating_hours=equip["hours"]
        )

        results.append({
            "equipment_id": equip["id"],
            "health_score": diagnosis["health_score"],
            "health_status": diagnosis["health_status"],
            "failure_prob": prediction["failure_probability"],
            "rul_days": prediction["rul_days"]
        })

        if diagnosis["health_status"] == "critical":
            critical_count += 1
        elif diagnosis["health_status"] == "at_risk":
            at_risk_count += 1

    # Print summary
    print("\n" + "-" * 70)
    print("FLEET HEALTH SUMMARY")
    print("-" * 70)
    print(f"{'Equipment':<12} {'Health':<8} {'Status':<12} {'Fail Prob':<12} {'RUL (days)':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['equipment_id']:<12} {r['health_score']:<8.1f} {r['health_status']:<12} "
              f"{r['failure_prob']:<12.2%} {r['rul_days']:<10.0f}")

    print("-" * 70)
    print(f"\nFleet Statistics:")
    print(f"  Total Equipment: {len(results)}")
    print(f"  Critical: {critical_count}")
    print(f"  At Risk: {at_risk_count}")
    print(f"  Healthy: {len(results) - critical_count - at_risk_count}")

    # Identify equipment needing immediate attention
    urgent = [r for r in results if r["health_status"] in ["critical", "at_risk"]]
    if urgent:
        print(f"\nEquipment Requiring Immediate Attention:")
        for r in urgent:
            print(f"  - {r['equipment_id']}: {r['health_status']} (Score: {r['health_score']:.1f})")

    return results


# =============================================================================
# EXAMPLE: ANOMALY DETECTION
# =============================================================================

async def example_anomaly_detection():
    """
    Example: Detect anomalies in sensor data streams.

    Demonstrates statistical anomaly detection for early warning.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Anomaly Detection")
    print("=" * 70)

    client = PredictMaintClient()

    # Example 1: Normal operation (no anomalies)
    print("\n--- Case 1: Normal Operation ---")
    normal_readings = {
        "vibration": [2.1, 2.2, 2.0, 2.3, 2.1, 2.2, 2.1, 2.0, 2.2, 2.1],
        "temperature": [55.0, 55.5, 54.8, 55.2, 55.1, 55.3, 55.0, 54.9, 55.2, 55.1],
        "pressure": [8.5, 8.4, 8.5, 8.6, 8.5, 8.4, 8.5, 8.5, 8.4, 8.5]
    }

    result = await client.detect_anomalies(
        equipment_id="PUMP-001",
        sensor_readings=normal_readings,
        sensitivity=3.0
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Anomaly Detected: {result['anomaly_detected']}")
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")
    print(f"Severity: {result['severity']}")

    # Example 2: Anomalous vibration
    print("\n--- Case 2: Anomalous Vibration ---")
    anomaly_readings = {
        "vibration": [2.1, 2.2, 2.0, 2.3, 2.1, 2.2, 2.1, 2.0, 2.2, 8.5],  # Spike!
        "temperature": [55.0, 55.5, 54.8, 55.2, 55.1, 55.3, 55.0, 54.9, 55.2, 55.1],
        "pressure": [8.5, 8.4, 8.5, 8.6, 8.5, 8.4, 8.5, 8.5, 8.4, 8.5]
    }

    result = await client.detect_anomalies(
        equipment_id="PUMP-002",
        sensor_readings=anomaly_readings,
        sensitivity=3.0
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Anomaly Detected: {result['anomaly_detected']}")
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")
    print(f"Severity: {result['severity']}")
    if result['contributing_factors']:
        print(f"Contributing Factors:")
        for factor in result['contributing_factors']:
            print(f"  - {factor['parameter']}: Z-score = {factor['z_score']:.2f}")

    # Example 3: Multiple anomalies
    print("\n--- Case 3: Multiple Anomalies ---")
    multi_anomaly_readings = {
        "vibration": [2.1, 2.2, 2.0, 2.3, 2.1, 2.2, 2.1, 2.0, 2.2, 9.5],
        "temperature": [55.0, 55.5, 54.8, 55.2, 55.1, 55.3, 55.0, 54.9, 55.2, 85.0],
        "pressure": [8.5, 8.4, 8.5, 8.6, 8.5, 8.4, 8.5, 8.5, 8.4, 8.5]
    }

    result = await client.detect_anomalies(
        equipment_id="PUMP-003",
        sensor_readings=multi_anomaly_readings,
        sensitivity=3.0
    )
    print(f"Equipment: {result['equipment_id']}")
    print(f"Anomaly Detected: {result['anomaly_detected']}")
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")
    print(f"Severity: {result['severity']}")
    print(f"Contributing Factors: {len(result['contributing_factors'])} parameters")
    for factor in result['contributing_factors']:
        print(f"  - {factor['parameter']}: value={factor['value']}, z={factor['z_score']:.2f}")

    return result


# =============================================================================
# EXAMPLE: COST ANALYSIS
# =============================================================================

async def example_cost_analysis():
    """
    Example: Perform maintenance cost analysis.

    Demonstrates cost-benefit analysis for maintenance decisions.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Cost Analysis")
    print("=" * 70)

    client = PredictMaintClient()

    # Define scenarios
    scenarios = [
        {
            "name": "Small Pump",
            "equipment_id": "PUMP-S01",
            "equipment_type": "pump",
            "hours": 20000,
            "health": 75,
            "pm_cost": 500,
            "failure_cost": 15000
        },
        {
            "name": "Large Motor",
            "equipment_id": "MOTOR-L01",
            "equipment_type": "motor",
            "hours": 30000,
            "health": 65,
            "pm_cost": 2500,
            "failure_cost": 85000
        },
        {
            "name": "Critical Compressor",
            "equipment_id": "COMP-C01",
            "equipment_type": "compressor",
            "hours": 40000,
            "health": 55,
            "pm_cost": 8000,
            "failure_cost": 250000
        }
    ]

    print(f"\nAnalyzing {len(scenarios)} maintenance scenarios...\n")

    total_savings = 0

    for scenario in scenarios:
        print(f"--- {scenario['name']} ({scenario['equipment_id']}) ---")

        # Get prediction
        prediction = await client.predict_failure(
            equipment_id=scenario["equipment_id"],
            equipment_type=scenario["equipment_type"],
            operating_hours=scenario["hours"]
        )

        # Calculate expected costs
        fail_prob = prediction["failure_probability"]
        expected_failure_cost = fail_prob * scenario["failure_cost"]
        expected_savings = expected_failure_cost - scenario["pm_cost"]

        print(f"  Current Hours: {scenario['hours']:,}")
        print(f"  Health Score: {scenario['health']}")
        print(f"  Failure Probability (1yr): {fail_prob:.2%}")
        print(f"  Preventive Maintenance Cost: ${scenario['pm_cost']:,.2f}")
        print(f"  Expected Failure Cost: ${expected_failure_cost:,.2f}")
        print(f"  Expected Savings: ${expected_savings:,.2f}")
        print(f"  ROI: {(expected_savings / scenario['pm_cost'] * 100) if scenario['pm_cost'] > 0 else 0:.1f}%")
        print()

        total_savings += max(0, expected_savings)

    print("-" * 50)
    print(f"Total Potential Savings: ${total_savings:,.2f}")

    return total_savings


# =============================================================================
# EXAMPLE: API RESPONSE FORMATS
# =============================================================================

async def example_api_responses():
    """
    Example: Demonstrate API response formats.

    Shows the structure of various API responses for documentation.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: API Response Formats")
    print("=" * 70)

    client = PredictMaintClient()

    # Diagnosis response
    print("\n--- Diagnosis Response Format ---")
    diagnosis = await client.diagnose_equipment(
        equipment_id="PUMP-001",
        equipment_type="pump",
        vibration_velocity_mm_s=3.5,
        temperature_c=62.0
    )
    print(json.dumps(diagnosis, indent=2, default=str))

    # Prediction response
    print("\n--- Prediction Response Format ---")
    prediction = await client.predict_failure(
        equipment_id="PUMP-001",
        equipment_type="pump",
        operating_hours=20000.0
    )
    print(json.dumps(prediction, indent=2, default=str))

    # Schedule response
    print("\n--- Schedule Response Format ---")
    schedule = await client.schedule_maintenance(
        equipment_id="PUMP-001",
        equipment_type="pump",
        current_health_score=75.0,
        operating_hours=20000.0
    )
    print(json.dumps(schedule, indent=2, default=str))

    # Anomaly response
    print("\n--- Anomaly Detection Response Format ---")
    anomaly = await client.detect_anomalies(
        equipment_id="PUMP-001",
        sensor_readings={
            "vibration": [2.0, 2.1, 2.2, 2.0, 2.1],
            "temperature": [55.0, 55.5, 55.2, 55.1, 55.3]
        }
    )
    print(json.dumps(anomaly, indent=2, default=str))


# =============================================================================
# EXAMPLE: ERROR HANDLING
# =============================================================================

async def example_error_handling():
    """
    Example: Demonstrate proper error handling.

    Shows how to handle various error conditions.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Error Handling")
    print("=" * 70)

    client = PredictMaintClient()

    # Example 1: Invalid equipment type
    print("\n--- Handling Invalid Equipment Type ---")
    try:
        # This would fail validation in the real API
        result = await client.diagnose_equipment(
            equipment_id="TEST-001",
            equipment_type="invalid_type",  # Invalid
            vibration_velocity_mm_s=2.5
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error handled: {type(e).__name__}: {e}")
        print("Recommendation: Use valid equipment types from EquipmentType enum")

    # Example 2: Missing required data
    print("\n--- Handling Missing Data ---")
    result = await client.diagnose_equipment(
        equipment_id="TEST-002",
        equipment_type="pump"
        # No sensor data provided
    )
    print(f"Health Score: {result['health_score']}")
    print("Note: Without sensor data, default health score is returned")

    # Example 3: Out of range values
    print("\n--- Handling Out of Range Values ---")
    try:
        result = await client.diagnose_equipment(
            equipment_id="TEST-003",
            equipment_type="pump",
            vibration_velocity_mm_s=1000.0  # Unrealistically high
        )
        print(f"Result processed (value may be clamped or flagged)")
    except Exception as e:
        print(f"Validation error: {e}")

    print("\n[Best Practices]")
    print("  1. Always validate inputs before API calls")
    print("  2. Handle exceptions gracefully")
    print("  3. Log errors for debugging")
    print("  4. Provide meaningful error messages to users")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main():
    """
    Run all example demonstrations.
    """
    print("\n")
    print("=" * 70)
    print("GL-013 PREDICTMAINT - Example Usage Demonstrations")
    print("=" * 70)
    print("\nThis script demonstrates the full capabilities of the")
    print("GL-013 Predictive Maintenance Agent.\n")

    # Run all examples
    try:
        await example_diagnose_equipment()
        await example_predict_failure()
        await example_schedule_maintenance()
        await example_full_workflow()
        await example_batch_processing()
        await example_anomaly_detection()
        await example_cost_analysis()
        await example_api_responses()
        await example_error_handling()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise


# =============================================================================
# QUICK START FUNCTIONS
# =============================================================================

def quick_start_diagnose():
    """Quick start: Run equipment diagnosis example."""
    asyncio.run(example_diagnose_equipment())


def quick_start_predict():
    """Quick start: Run failure prediction example."""
    asyncio.run(example_predict_failure())


def quick_start_schedule():
    """Quick start: Run maintenance scheduling example."""
    asyncio.run(example_schedule_maintenance())


def quick_start_workflow():
    """Quick start: Run full workflow example."""
    asyncio.run(example_full_workflow())


def quick_start_batch():
    """Quick start: Run batch processing example."""
    asyncio.run(example_batch_processing())


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run all examples
    asyncio.run(main())
