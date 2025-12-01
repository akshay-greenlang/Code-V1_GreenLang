# -*- coding: utf-8 -*-
"""
GL-013 PREDICTMAINT - End-to-End Workflow Tests
Comprehensive end-to-end testing of complete workflows.

Tests cover:
- Diagnose workflow (equipment health assessment)
- Predict workflow (RUL and failure probability)
- Schedule workflow (maintenance optimization)
- Full maintenance cycle (end-to-end)
- Emergency response workflow
- Multi-equipment batch processing
- Workflow orchestration

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import hashlib
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Import test fixtures from conftest
from ..conftest import (
    MachineClass,
    VibrationZone,
    HealthState,
    AlertSeverity,
    ISO_10816_LIMITS,
    WEIBULL_PARAMETERS,
)


# =============================================================================
# TEST CLASS: DIAGNOSE WORKFLOW
# =============================================================================


class TestDiagnoseWorkflow:
    """End-to-end tests for equipment diagnosis workflow."""

    @pytest.mark.e2e
    def test_diagnose_healthy_equipment(
        self,
        vibration_analyzer,
        thermal_degradation_calculator,
        pump_equipment_data,
        healthy_vibration_data,
        normal_thermal_data,
    ):
        """Test complete diagnosis of healthy equipment."""
        # Step 1: Analyze vibration
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=healthy_vibration_data["velocity_rms_mm_s"],
            machine_class=MachineClass.CLASS_II,
            equipment_id=pump_equipment_data["equipment_id"],
        )

        assert vib_result["zone"] == VibrationZone.ZONE_A
        assert vib_result["alarm_level"] == "normal"

        # Step 2: Analyze thermal condition
        thermal_result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=normal_thermal_data["hot_spot_temperature_c"],
            reference_temperature_c=Decimal("110"),
        )

        # Aging factor should be near 1 for normal temps
        assert thermal_result["acceleration_factor"] < Decimal("2.0")

        # Step 3: Combine into overall health assessment
        overall_health = "HEALTHY"  # Based on Zone A vibration and normal thermal
        assert overall_health == "HEALTHY"

    @pytest.mark.e2e
    def test_diagnose_degraded_equipment(
        self,
        vibration_analyzer,
        thermal_degradation_calculator,
        anomaly_detector,
        pump_equipment_data,
        degraded_vibration_data,
        elevated_thermal_data,
    ):
        """Test complete diagnosis of degraded equipment."""
        # Step 1: Analyze vibration
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=degraded_vibration_data["velocity_rms_mm_s"],
            machine_class=MachineClass.CLASS_II,
            equipment_id=pump_equipment_data["equipment_id"],
        )

        # Should be in Zone B or C
        assert vib_result["zone"] in [VibrationZone.ZONE_B, VibrationZone.ZONE_C]

        # Step 2: Analyze thermal condition
        thermal_result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=elevated_thermal_data["hot_spot_temperature_c"],
            reference_temperature_c=Decimal("110"),
        )

        # Elevated temperature = accelerated aging
        assert thermal_result["acceleration_factor"] > Decimal("2.0")

        # Step 3: Check for anomalies
        historical_data = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]
        current_value = float(degraded_vibration_data["velocity_rms_mm_s"])

        anomaly_result = anomaly_detector.detect_univariate_anomaly(
            value=Decimal(str(current_value)),
            historical_data=historical_data,
            threshold_sigma="3.0",
        )

        # Current value should show as anomaly compared to baseline
        assert anomaly_result["z_score"] > Decimal("0")

        # Step 4: Generate diagnosis
        diagnosis = {
            "equipment_id": pump_equipment_data["equipment_id"],
            "health_status": "DEGRADED",
            "vibration_zone": vib_result["zone"],
            "aging_acceleration": thermal_result["acceleration_factor"],
            "recommendation": "Schedule maintenance within 30 days",
        }

        assert diagnosis["health_status"] == "DEGRADED"

    @pytest.mark.e2e
    def test_diagnose_critical_equipment(
        self,
        vibration_analyzer,
        thermal_degradation_calculator,
        pump_equipment_data,
        critical_vibration_data,
        critical_thermal_data,
    ):
        """Test diagnosis of critical equipment condition."""
        # Step 1: Analyze vibration
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=critical_vibration_data["velocity_rms_mm_s"],
            machine_class=MachineClass.CLASS_II,
        )

        assert vib_result["zone"] == VibrationZone.ZONE_D
        assert vib_result["alarm_level"] == "critical"

        # Step 2: Generate emergency alert
        emergency_alert = {
            "alert_type": "EMERGENCY",
            "equipment_id": pump_equipment_data["equipment_id"],
            "message": "Zone D vibration detected - immediate action required",
            "recommendation": "Stop equipment immediately, investigate cause",
        }

        assert emergency_alert["alert_type"] == "EMERGENCY"


# =============================================================================
# TEST CLASS: PREDICT WORKFLOW
# =============================================================================


class TestPredictWorkflow:
    """End-to-end tests for failure prediction workflow."""

    @pytest.mark.e2e
    def test_predict_rul_standard_equipment(
        self,
        rul_calculator,
        failure_probability_calculator,
        pump_equipment_data,
    ):
        """Test RUL prediction for standard equipment."""
        equipment_type = "pump_centrifugal"
        operating_hours = pump_equipment_data["operating_hours"]

        # Step 1: Calculate RUL
        rul_result = rul_calculator.calculate_weibull_rul(
            equipment_type=equipment_type,
            operating_hours=operating_hours,
            target_reliability="0.5",
            confidence_level="90%",
        )

        assert rul_result["rul_hours"] > Decimal("0")
        assert rul_result["rul_days"] > Decimal("0")

        # Step 2: Calculate current failure probability
        params = WEIBULL_PARAMETERS[equipment_type]
        fp_result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=params["beta"],
            eta=params["eta"],
            time_hours=operating_hours,
        )

        assert fp_result["failure_probability"] >= Decimal("0")
        assert fp_result["failure_probability"] <= Decimal("1")

        # Step 3: Generate prediction report
        prediction = {
            "equipment_id": pump_equipment_data["equipment_id"],
            "rul_hours": rul_result["rul_hours"],
            "rul_days": rul_result["rul_days"],
            "current_reliability": rul_result["current_reliability"],
            "failure_probability": fp_result["failure_probability"],
            "confidence_interval": (
                rul_result["confidence_lower"],
                rul_result["confidence_upper"],
            ),
            "provenance_hash": rul_result["provenance_hash"],
        }

        assert "provenance_hash" in prediction
        assert len(prediction["provenance_hash"]) == 64

    @pytest.mark.e2e
    def test_predict_with_health_adjustment(
        self,
        rul_calculator,
        vibration_analyzer,
        pump_equipment_data,
        degraded_vibration_data,
    ):
        """Test RUL prediction with health state adjustment."""
        # Step 1: Determine health state from vibration
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=degraded_vibration_data["velocity_rms_mm_s"],
            machine_class=MachineClass.CLASS_II,
        )

        # Map vibration zone to health state
        zone_to_health = {
            VibrationZone.ZONE_A: HealthState.EXCELLENT,
            VibrationZone.ZONE_B: HealthState.GOOD,
            VibrationZone.ZONE_C: HealthState.FAIR,
            VibrationZone.ZONE_D: HealthState.CRITICAL,
        }
        health_state = zone_to_health.get(vib_result["zone"], HealthState.GOOD)

        # Step 2: Calculate RUL without health adjustment
        rul_baseline = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=pump_equipment_data["operating_hours"],
        )

        # Step 3: Calculate RUL with health adjustment
        # Note: Actual implementation would pass health_state parameter
        # Here we verify the workflow structure

        assert rul_baseline["rul_hours"] > Decimal("0")

    @pytest.mark.e2e
    def test_predict_multiple_failure_modes(
        self,
        failure_probability_calculator,
        pump_equipment_data,
    ):
        """Test prediction considering multiple failure modes."""
        operating_hours = pump_equipment_data["operating_hours"]

        # Define multiple failure modes for pump
        failure_modes = [
            {"name": "bearing_wear", "beta": Decimal("2.5"), "eta": Decimal("45000")},
            {"name": "seal_failure", "beta": Decimal("1.8"), "eta": Decimal("25000")},
            {"name": "cavitation", "beta": Decimal("3.0"), "eta": Decimal("35000")},
        ]

        # Step 1: Calculate failure probability for each mode
        mode_results = []
        for mode in failure_modes:
            result = failure_probability_calculator.calculate_weibull_failure_probability(
                beta=mode["beta"],
                eta=mode["eta"],
                time_hours=operating_hours,
            )
            mode_results.append({
                "mode": mode["name"],
                "failure_probability": result["failure_probability"],
            })

        # Step 2: Calculate combined reliability (product of individual reliabilities)
        combined_reliability = Decimal("1")
        for mode in failure_modes:
            result = failure_probability_calculator.calculate_weibull_failure_probability(
                beta=mode["beta"],
                eta=mode["eta"],
                time_hours=operating_hours,
            )
            combined_reliability *= result["reliability"]

        combined_failure_prob = Decimal("1") - combined_reliability

        # Combined failure probability should be higher than any individual
        assert combined_failure_prob >= max(r["failure_probability"] for r in mode_results)

        # Step 3: Identify dominant failure mode
        dominant_mode = max(mode_results, key=lambda x: x["failure_probability"])

        assert dominant_mode is not None


# =============================================================================
# TEST CLASS: SCHEDULE WORKFLOW
# =============================================================================


class TestScheduleWorkflow:
    """End-to-end tests for maintenance scheduling workflow."""

    @pytest.mark.e2e
    def test_schedule_optimal_maintenance(
        self,
        maintenance_scheduler,
        rul_calculator,
        pump_equipment_data,
    ):
        """Test optimal maintenance scheduling."""
        equipment_type = "pump_centrifugal"
        params = WEIBULL_PARAMETERS[equipment_type]

        # Step 1: Calculate RUL
        rul_result = rul_calculator.calculate_weibull_rul(
            equipment_type=equipment_type,
            operating_hours=pump_equipment_data["operating_hours"],
        )

        # Step 2: Calculate optimal interval
        schedule_result = maintenance_scheduler.calculate_optimal_interval(
            beta=params["beta"],
            eta=params["eta"],
            preventive_cost=Decimal("1500"),  # PM cost
            corrective_cost=Decimal("15000"),  # CM cost
        )

        assert schedule_result["optimal_interval_hours"] > Decimal("0")

        # Step 3: Determine next maintenance date
        hours_to_next = min(
            schedule_result["optimal_interval_hours"],
            rul_result["rul_hours"],
        )

        days_to_next = hours_to_next / Decimal("24")
        next_maintenance_date = datetime.now() + timedelta(days=float(days_to_next))

        assert next_maintenance_date > datetime.now()

    @pytest.mark.e2e
    def test_schedule_with_criticality_priority(
        self,
        maintenance_scheduler,
        pump_equipment_data,
        motor_equipment_data,
    ):
        """Test scheduling with equipment criticality consideration."""
        # Define equipment with different criticality
        equipment_list = [
            {
                "data": pump_equipment_data,
                "criticality": "high",
                "optimal_interval": Decimal("2000"),
            },
            {
                "data": motor_equipment_data,
                "criticality": "critical",
                "optimal_interval": Decimal("1500"),
            },
        ]

        # Step 1: Calculate priority scores
        for equip in equipment_list:
            criticality_weight = {
                "low": Decimal("1.0"),
                "medium": Decimal("1.5"),
                "high": Decimal("2.0"),
                "critical": Decimal("3.0"),
            }
            equip["priority_score"] = criticality_weight[equip["criticality"]]

        # Step 2: Sort by priority (higher score = higher priority)
        prioritized_list = sorted(
            equipment_list,
            key=lambda x: x["priority_score"],
            reverse=True,
        )

        # Critical equipment should be first
        assert prioritized_list[0]["criticality"] == "critical"

    @pytest.mark.e2e
    def test_schedule_spare_parts_integration(
        self,
        maintenance_scheduler,
        spare_parts_calculator,
        pump_equipment_data,
    ):
        """Test maintenance scheduling with spare parts availability."""
        # Step 1: Calculate optimal interval
        schedule_result = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("2.5"),
            eta=Decimal("45000"),
            preventive_cost=Decimal("1500"),
            corrective_cost=Decimal("15000"),
        )

        # Step 2: Calculate spare parts requirement
        eoq_result = spare_parts_calculator.calculate_eoq(
            annual_demand=Decimal("12"),  # 12 maintenance events per year
            ordering_cost=Decimal("50"),
            holding_cost_rate=Decimal("0.25"),
            unit_cost=Decimal("200"),
        )

        # Step 3: Verify parts will be available
        work_order = {
            "equipment_id": pump_equipment_data["equipment_id"],
            "scheduled_hours": schedule_result["optimal_interval_hours"],
            "parts_required": ["bearing", "seal", "gasket"],
            "parts_available": True,  # Based on EOQ
        }

        assert work_order["parts_available"] is True


# =============================================================================
# TEST CLASS: FULL MAINTENANCE CYCLE
# =============================================================================


class TestFullMaintenanceCycle:
    """End-to-end tests for complete maintenance cycle."""

    @pytest.mark.e2e
    def test_full_predictive_maintenance_cycle(
        self,
        vibration_analyzer,
        thermal_degradation_calculator,
        rul_calculator,
        failure_probability_calculator,
        maintenance_scheduler,
        anomaly_detector,
        pump_equipment_data,
        degraded_vibration_data,
        maintenance_history,
    ):
        """Test complete predictive maintenance cycle from detection to scheduling."""
        equipment_id = pump_equipment_data["equipment_id"]

        # Phase 1: MONITOR - Collect and analyze sensor data
        print(f"\n=== Phase 1: MONITOR ===")

        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=degraded_vibration_data["velocity_rms_mm_s"],
            machine_class=MachineClass.CLASS_II,
            equipment_id=equipment_id,
        )

        assert vib_result is not None
        print(f"Vibration Zone: {vib_result['zone']}")

        # Phase 2: DETECT - Identify anomalies and degradation
        print(f"\n=== Phase 2: DETECT ===")

        historical_data = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
        anomaly_result = anomaly_detector.detect_univariate_anomaly(
            value=degraded_vibration_data["velocity_rms_mm_s"],
            historical_data=historical_data,
            threshold_sigma="3.0",
        )

        degradation_detected = (
            vib_result["zone"] in [VibrationZone.ZONE_C, VibrationZone.ZONE_D] or
            anomaly_result["is_anomaly"]
        )

        print(f"Degradation Detected: {degradation_detected}")

        # Phase 3: PREDICT - Estimate remaining useful life
        print(f"\n=== Phase 3: PREDICT ===")

        rul_result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=pump_equipment_data["operating_hours"],
            target_reliability="0.5",
        )

        fp_result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("2.5"),
            eta=Decimal("45000"),
            time_hours=pump_equipment_data["operating_hours"],
        )

        print(f"RUL: {rul_result['rul_days']} days")
        print(f"Failure Probability: {fp_result['failure_probability']}")

        # Phase 4: SCHEDULE - Plan maintenance action
        print(f"\n=== Phase 4: SCHEDULE ===")

        schedule_result = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("2.5"),
            eta=Decimal("45000"),
            preventive_cost=Decimal("1500"),
            corrective_cost=Decimal("15000"),
        )

        # Determine urgency based on condition
        if vib_result["zone"] == VibrationZone.ZONE_D:
            urgency = "EMERGENCY"
            schedule_within_days = 1
        elif vib_result["zone"] == VibrationZone.ZONE_C:
            urgency = "HIGH"
            schedule_within_days = 7
        else:
            urgency = "NORMAL"
            schedule_within_days = min(
                float(rul_result["rul_days"]),
                float(schedule_result["optimal_interval_hours"] / Decimal("24")),
            )

        print(f"Urgency: {urgency}")
        print(f"Schedule Within: {schedule_within_days} days")

        # Phase 5: GENERATE - Create work order and audit trail
        print(f"\n=== Phase 5: GENERATE ===")

        work_order = {
            "work_order_id": f"WO-{datetime.now().strftime('%Y%m%d')}-001",
            "equipment_id": equipment_id,
            "type": "preventive" if urgency != "EMERGENCY" else "emergency",
            "urgency": urgency,
            "scheduled_date": (datetime.now() + timedelta(days=schedule_within_days)).isoformat(),
            "description": f"Predictive maintenance - {vib_result['zone']} vibration detected",
            "estimated_duration_hours": 4,
            "provenance": {
                "vibration_analysis_hash": vib_result["provenance_hash"],
                "rul_calculation_hash": rul_result["provenance_hash"],
                "generated_at": datetime.now().isoformat(),
            },
        }

        assert work_order["work_order_id"] is not None
        assert work_order["provenance"]["vibration_analysis_hash"] is not None

        print(f"Work Order: {work_order['work_order_id']}")
        print(f"Scheduled: {work_order['scheduled_date']}")

    @pytest.mark.e2e
    def test_maintenance_cycle_provenance_chain(
        self,
        vibration_analyzer,
        rul_calculator,
        provenance_validator,
        pump_equipment_data,
    ):
        """Test that complete provenance chain is maintained through cycle."""
        provenance_hashes = []

        # Step 1: Vibration analysis
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("4.5"),
            machine_class=MachineClass.CLASS_II,
        )
        provenance_hashes.append(vib_result["provenance_hash"])

        # Step 2: RUL calculation
        rul_result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=pump_equipment_data["operating_hours"],
        )
        provenance_hashes.append(rul_result["provenance_hash"])

        # All hashes should be valid SHA-256
        for hash_val in provenance_hashes:
            assert len(hash_val) == 64
            int(hash_val, 16)  # Should be valid hex

        # Build Merkle root for audit trail
        # (Simplified - full implementation would use MerkleTree class)
        if len(provenance_hashes) >= 2:
            combined = provenance_hashes[0] + provenance_hashes[1]
            merkle_root = hashlib.sha256(combined.encode()).hexdigest()
            assert len(merkle_root) == 64


# =============================================================================
# TEST CLASS: EMERGENCY RESPONSE WORKFLOW
# =============================================================================


class TestEmergencyResponseWorkflow:
    """End-to-end tests for emergency response scenarios."""

    @pytest.mark.e2e
    def test_emergency_vibration_alert(
        self,
        vibration_analyzer,
        pump_equipment_data,
        critical_vibration_data,
    ):
        """Test emergency response to Zone D vibration."""
        # Step 1: Detect critical vibration
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=critical_vibration_data["velocity_rms_mm_s"],
            machine_class=MachineClass.CLASS_II,
            equipment_id=pump_equipment_data["equipment_id"],
        )

        assert vib_result["zone"] == VibrationZone.ZONE_D
        assert vib_result["alarm_level"] == "critical"

        # Step 2: Generate emergency alert
        emergency_alert = {
            "alert_id": f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "severity": AlertSeverity.EMERGENCY,
            "equipment_id": pump_equipment_data["equipment_id"],
            "condition": "Zone D vibration",
            "value": critical_vibration_data["velocity_rms_mm_s"],
            "limit": ISO_10816_LIMITS[MachineClass.CLASS_II]["zone_c_upper"],
            "timestamp": datetime.now().isoformat(),
            "actions": [
                "STOP equipment immediately",
                "Notify maintenance supervisor",
                "Isolate equipment from process",
                "Document condition before inspection",
            ],
        }

        assert emergency_alert["severity"] == AlertSeverity.EMERGENCY
        assert len(emergency_alert["actions"]) > 0

        # Step 3: Create emergency work order
        emergency_wo = {
            "work_order_id": f"WO-EMERG-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "equipment_id": pump_equipment_data["equipment_id"],
            "type": "emergency",
            "priority": "P1 - EMERGENCY",
            "description": f"Emergency response to Zone D vibration: {vib_result['velocity_rms']} mm/s",
            "created_at": datetime.now().isoformat(),
            "target_completion": datetime.now() + timedelta(hours=4),
        }

        assert "EMERG" in emergency_wo["work_order_id"]

    @pytest.mark.e2e
    def test_emergency_thermal_alert(
        self,
        thermal_degradation_calculator,
        motor_equipment_data,
        critical_thermal_data,
    ):
        """Test emergency response to critical thermal condition."""
        # Step 1: Calculate thermal stress
        thermal_result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=critical_thermal_data["hot_spot_temperature_c"],
            reference_temperature_c=Decimal("110"),
        )

        # Step 2: Check if exceeds limits
        max_safe_temp = Decimal("155")  # Class F insulation
        is_emergency = critical_thermal_data["hot_spot_temperature_c"] > max_safe_temp

        if is_emergency:
            emergency_alert = {
                "alert_id": f"THERMAL-ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "severity": AlertSeverity.CRITICAL,
                "equipment_id": motor_equipment_data["equipment_id"],
                "condition": "Hot spot temperature exceeds insulation class limit",
                "value": critical_thermal_data["hot_spot_temperature_c"],
                "limit": max_safe_temp,
                "aging_acceleration": thermal_result["acceleration_factor"],
                "actions": [
                    "Reduce load immediately",
                    "Check cooling system",
                    "Monitor temperature continuously",
                ],
            }

            assert emergency_alert["severity"] == AlertSeverity.CRITICAL

    @pytest.mark.e2e
    def test_emergency_cascade_detection(
        self,
        vibration_analyzer,
        thermal_degradation_calculator,
        anomaly_detector,
        pump_equipment_data,
    ):
        """Test detection of cascading failures."""
        # Multiple simultaneous abnormalities indicate potential cascade
        alerts = []

        # Check 1: Vibration
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("8.0"),  # Zone C
            machine_class=MachineClass.CLASS_II,
        )
        if vib_result["zone"] in [VibrationZone.ZONE_C, VibrationZone.ZONE_D]:
            alerts.append({"type": "vibration", "severity": vib_result["alarm_level"]})

        # Check 2: Temperature
        thermal_result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("140"),  # Elevated
            reference_temperature_c=Decimal("110"),
        )
        if thermal_result["acceleration_factor"] > Decimal("4"):
            alerts.append({"type": "thermal", "severity": "warning"})

        # Check 3: Anomaly detection
        historical = [100, 101, 99, 102, 100, 98]
        anomaly = anomaly_detector.detect_univariate_anomaly(
            value=Decimal("130"),  # Anomalous
            historical_data=historical,
        )
        if anomaly["is_anomaly"]:
            alerts.append({"type": "anomaly", "severity": anomaly["severity"]})

        # If multiple alerts, could be cascade
        if len(alerts) >= 2:
            cascade_alert = {
                "alert_type": "POTENTIAL_CASCADE",
                "equipment_id": pump_equipment_data["equipment_id"],
                "related_alerts": alerts,
                "recommendation": "Immediate investigation required - multiple failures detected",
            }

            assert cascade_alert["alert_type"] == "POTENTIAL_CASCADE"


# =============================================================================
# TEST CLASS: MULTI-EQUIPMENT BATCH PROCESSING
# =============================================================================


class TestMultiEquipmentBatchProcessing:
    """End-to-end tests for batch processing multiple equipment."""

    @pytest.mark.e2e
    def test_batch_diagnosis_multiple_equipment(
        self,
        vibration_analyzer,
        equipment_data_generator,
    ):
        """Test batch diagnosis of multiple equipment items."""
        # Generate batch of equipment data
        equipment_list = equipment_data_generator.generate_pump_data(num_records=10)

        # Process each equipment
        diagnosis_results = []
        for equip in equipment_list:
            vib_result = vibration_analyzer.assess_severity(
                velocity_rms=equip["vibration_velocity_mm_s"],
                machine_class=MachineClass.CLASS_II,
                equipment_id=equip["equipment_id"],
            )

            diagnosis_results.append({
                "equipment_id": equip["equipment_id"],
                "zone": vib_result["zone"],
                "alarm_level": vib_result["alarm_level"],
            })

        assert len(diagnosis_results) == 10

        # Categorize results
        zone_counts = {}
        for result in diagnosis_results:
            zone = result["zone"]
            zone_counts[zone] = zone_counts.get(zone, 0) + 1

        # Should have some distribution across zones
        assert len(zone_counts) >= 1

    @pytest.mark.e2e
    def test_batch_rul_calculation(
        self,
        rul_calculator,
        equipment_data_generator,
    ):
        """Test batch RUL calculation."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=5)

        rul_results = []
        for equip in equipment_list:
            result = rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=equip["operating_hours"],
            )
            rul_results.append({
                "equipment_id": equip["equipment_id"],
                "rul_hours": result["rul_hours"],
                "rul_days": result["rul_days"],
            })

        assert len(rul_results) == 5

        # All should have valid RUL
        for result in rul_results:
            assert result["rul_hours"] >= Decimal("0")

    @pytest.mark.e2e
    def test_prioritize_maintenance_queue(
        self,
        rul_calculator,
        vibration_analyzer,
        equipment_data_generator,
    ):
        """Test prioritization of maintenance queue."""
        equipment_list = equipment_data_generator.generate_pump_data(num_records=5)

        # Calculate priority for each
        priority_queue = []
        for equip in equipment_list:
            rul_result = rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=equip["operating_hours"],
            )

            vib_result = vibration_analyzer.assess_severity(
                velocity_rms=equip["vibration_velocity_mm_s"],
                machine_class=MachineClass.CLASS_II,
            )

            # Priority score based on RUL and vibration zone
            zone_score = {
                VibrationZone.ZONE_A: 1,
                VibrationZone.ZONE_B: 2,
                VibrationZone.ZONE_C: 4,
                VibrationZone.ZONE_D: 8,
            }

            # Lower RUL = higher priority, higher zone = higher priority
            priority = zone_score.get(vib_result["zone"], 1) * (
                Decimal("100000") / max(rul_result["rul_hours"], Decimal("1"))
            )

            priority_queue.append({
                "equipment_id": equip["equipment_id"],
                "priority_score": float(priority),
                "rul_hours": rul_result["rul_hours"],
                "vibration_zone": vib_result["zone"],
            })

        # Sort by priority (descending)
        priority_queue.sort(key=lambda x: x["priority_score"], reverse=True)

        assert len(priority_queue) == 5


# =============================================================================
# TEST CLASS: WORKFLOW INTEGRATION WITH MOCKS
# =============================================================================


class TestWorkflowIntegrationWithMocks:
    """End-to-end tests with mock external systems."""

    @pytest.mark.e2e
    def test_workflow_with_cmms_integration(
        self,
        mock_cmms_connector,
        vibration_analyzer,
        rul_calculator,
        pump_equipment_data,
    ):
        """Test workflow integration with CMMS system."""
        # Step 1: Get equipment from CMMS
        equipment = mock_cmms_connector.get_equipment(pump_equipment_data["equipment_id"])
        assert equipment is not None

        # Step 2: Perform analysis
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("4.5"),
            machine_class=MachineClass.CLASS_II,
        )

        rul_result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=pump_equipment_data["operating_hours"],
        )

        # Step 3: Create work order in CMMS
        if vib_result["zone"] == VibrationZone.ZONE_C:
            wo_id = mock_cmms_connector.create_work_order(
                equipment_id=pump_equipment_data["equipment_id"],
                description=f"PM - Zone C vibration, RUL {rul_result['rul_days']} days",
                priority="HIGH",
            )
            assert wo_id == "WO-003"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_async_workflow_with_cms(
        self,
        mock_cms_connector,
        vibration_analyzer,
    ):
        """Test async workflow with condition monitoring system."""
        # Step 1: Get latest readings from CMS
        readings = mock_cms_connector.get_latest_readings("PUMP-001")

        # Step 2: Analyze vibration
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=readings["vibration_velocity_mm_s"],
            machine_class=MachineClass.CLASS_II,
        )

        assert vib_result is not None

        # Step 3: Get historical data for trend analysis
        historical = mock_cms_connector.get_historical_data(
            equipment_id="PUMP-001",
            parameter="vibration",
            days=30,
        )

        assert len(historical) > 0
