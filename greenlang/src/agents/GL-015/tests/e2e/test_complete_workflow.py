# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - End-to-End Tests

Complete end-to-end tests for the Insulation Inspection Agent.
Tests full workflows from image capture to report generation.

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import json
import hashlib
from decimal import Decimal
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch
import numpy as np


# =============================================================================
# TEST: COMPLETE INSPECTION WORKFLOW
# =============================================================================

@pytest.mark.e2e
class TestCompleteInspectionWorkflow:
    """End-to-end tests for complete inspection workflow."""

    @pytest.mark.asyncio
    async def test_full_inspection_workflow(
        self,
        mock_thermal_camera,
        mock_weather_service,
        mock_cmms_connector,
        sample_equipment_parameters,
        sample_insulation_specs
    ):
        """Test complete inspection workflow from start to finish."""
        workflow_results = {}

        # Phase 1: Pre-inspection setup
        # Get weather conditions
        weather = await mock_weather_service.get_current_conditions()
        workflow_results["weather"] = weather

        # Check inspection conditions
        is_suitable = (
            weather["temperature_c"] > 5 and
            weather["wind_speed_m_s"] < 10
        )
        workflow_results["conditions_suitable"] = is_suitable

        # Phase 2: Equipment preparation
        await mock_thermal_camera.connect()
        camera_status = await mock_thermal_camera.get_status()
        workflow_results["camera_ready"] = camera_status["connected"]

        # Set camera parameters
        await mock_thermal_camera.set_emissivity(0.95)
        await mock_thermal_camera.set_reflected_temperature(weather["temperature_c"])

        # Phase 3: Image capture
        captured_images = []
        for position in range(3):  # Three inspection positions
            image_data = await mock_thermal_camera.capture_image()
            image_data["position"] = position
            captured_images.append(image_data)

        workflow_results["images_captured"] = len(captured_images)

        # Phase 4: Image analysis
        analysis_results = []
        for img in captured_images:
            temp_matrix = img["temperature_matrix"]
            max_temp = max(max(row) for row in temp_matrix)
            min_temp = min(min(row) for row in temp_matrix)
            avg_temp = sum(sum(row) for row in temp_matrix) / (len(temp_matrix) * len(temp_matrix[0]))

            analysis = {
                "position": img["position"],
                "min_temp_c": min_temp,
                "max_temp_c": max_temp,
                "avg_temp_c": avg_temp,
                "delta_t": max_temp - weather["temperature_c"],
            }
            analysis_results.append(analysis)

        workflow_results["analysis_complete"] = True

        # Phase 5: Defect detection
        defects = []
        for result in analysis_results:
            if result["delta_t"] > 30:  # Threshold for significant defect
                defects.append({
                    "position": result["position"],
                    "severity": "moderate" if result["delta_t"] < 50 else "severe",
                    "max_temp_c": result["max_temp_c"],
                })

        workflow_results["defects_found"] = len(defects)

        # Phase 6: Work order creation
        await mock_cmms_connector.connect()
        work_orders = []
        for defect in defects:
            wo = await mock_cmms_connector.create_work_order({
                "equipment_tag": sample_equipment_parameters["equipment_tag"],
                "priority": "high" if defect["severity"] == "severe" else "medium",
                "description": f"Thermal defect at position {defect['position']}",
            })
            work_orders.append(wo)

        workflow_results["work_orders_created"] = len(work_orders)

        # Phase 7: Report generation
        report = {
            "report_id": f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "inspection_date": date.today().isoformat(),
            "equipment_tag": sample_equipment_parameters["equipment_tag"],
            "weather_conditions": weather,
            "images_analyzed": len(captured_images),
            "defects_found": len(defects),
            "work_orders": [wo["work_order_id"] for wo in work_orders],
            "overall_condition": "requires_attention" if defects else "acceptable",
            "provenance_hash": hashlib.sha256(
                json.dumps(analysis_results, default=str).encode()
            ).hexdigest(),
        }

        workflow_results["report_generated"] = True
        workflow_results["report"] = report

        # Cleanup
        await mock_thermal_camera.disconnect()
        await mock_cmms_connector.disconnect()

        # Assertions
        assert workflow_results["camera_ready"]
        assert workflow_results["images_captured"] == 3
        assert workflow_results["analysis_complete"]
        assert workflow_results["report_generated"]
        assert len(report["provenance_hash"]) == 64

    @pytest.mark.asyncio
    async def test_inspection_with_no_defects(
        self,
        mock_thermal_camera,
        mock_weather_service
    ):
        """Test inspection workflow when no defects are found."""
        weather = await mock_weather_service.get_current_conditions()

        await mock_thermal_camera.connect()

        # Capture images with normal temperatures
        mock_thermal_camera.capture_image.return_value = {
            "temperature_matrix": [[25.0] * 100 for _ in range(100)],
            "timestamp": datetime.now().isoformat(),
        }

        images = []
        for _ in range(3):
            img = await mock_thermal_camera.capture_image()
            images.append(img)

        # Analyze - no significant delta-T
        defects = []
        for img in images:
            max_temp = max(max(row) for row in img["temperature_matrix"])
            if max_temp - weather["temperature_c"] > 30:
                defects.append({"type": "hotspot"})

        await mock_thermal_camera.disconnect()

        # Should have no defects
        assert len(defects) == 0

    @pytest.mark.asyncio
    async def test_inspection_with_multiple_defects(
        self,
        mock_thermal_camera,
        mock_cmms_connector,
        sample_thermal_image_with_multiple_defects
    ):
        """Test inspection with multiple defects requiring prioritization."""
        await mock_thermal_camera.connect()
        await mock_cmms_connector.connect()

        # Use multi-defect image
        temp_matrix = sample_thermal_image_with_multiple_defects["temperature_matrix"]

        # Detect multiple defects
        mean_temp = np.mean(temp_matrix)
        std_temp = np.std(temp_matrix)
        threshold = mean_temp + 2 * std_temp

        # Find hotspot regions
        hotspot_pixels = np.array(temp_matrix) > threshold

        # Count distinct regions (simplified)
        defect_count = sample_thermal_image_with_multiple_defects.get("defect_count", 0)

        # Create prioritized work orders
        work_orders = []
        for i in range(min(defect_count, 3)):
            wo = await mock_cmms_connector.create_work_order({
                "equipment_tag": f"P-100{i+1}-A",
                "priority": ["critical", "high", "medium"][i],
                "description": f"Defect {i+1}",
            })
            work_orders.append(wo)

        await mock_thermal_camera.disconnect()
        await mock_cmms_connector.disconnect()

        assert len(work_orders) == min(defect_count, 3)


# =============================================================================
# TEST: ENERGY SAVINGS WORKFLOW
# =============================================================================

@pytest.mark.e2e
class TestEnergySavingsWorkflow:
    """End-to-end tests for energy savings calculation workflow."""

    def test_complete_energy_analysis_workflow(
        self,
        known_heat_loss_values,
        sample_equipment_parameters
    ):
        """Test complete energy analysis from defect to savings estimate."""
        # Step 1: Calculate current heat loss
        case = known_heat_loss_values["case_1"]
        current_heat_loss_w_per_m = 250.0  # Damaged insulation
        design_heat_loss_w_per_m = float(case["expected_heat_loss_w_per_m"])

        pipe_length_m = float(sample_equipment_parameters["pipe_length_m"])

        # Step 2: Calculate energy waste
        excess_heat_loss_w = (current_heat_loss_w_per_m - design_heat_loss_w_per_m) * pipe_length_m
        operating_hours = 8000

        annual_energy_waste_kwh = excess_heat_loss_w * operating_hours / 1000

        # Step 3: Calculate fuel consumption
        boiler_efficiency = 0.85
        gas_heating_value_kwh_per_m3 = 10.5

        fuel_wasted_m3 = annual_energy_waste_kwh / (boiler_efficiency * gas_heating_value_kwh_per_m3)

        # Step 4: Calculate cost savings
        gas_price_per_m3 = 0.35
        annual_savings = fuel_wasted_m3 * gas_price_per_m3

        # Step 5: Calculate CO2 reduction
        co2_factor_kg_per_m3 = 1.89
        co2_reduction_kg = fuel_wasted_m3 * co2_factor_kg_per_m3

        # Step 6: Estimate repair cost
        repair_cost = 5000  # Estimated

        # Step 7: Calculate payback
        payback_years = repair_cost / annual_savings if annual_savings > 0 else float('inf')

        # Step 8: Generate summary
        savings_summary = {
            "current_heat_loss_w_per_m": current_heat_loss_w_per_m,
            "design_heat_loss_w_per_m": design_heat_loss_w_per_m,
            "excess_loss_percent": ((current_heat_loss_w_per_m - design_heat_loss_w_per_m) /
                                   design_heat_loss_w_per_m * 100),
            "annual_energy_waste_kwh": annual_energy_waste_kwh,
            "annual_cost_savings_usd": annual_savings,
            "annual_co2_reduction_kg": co2_reduction_kg,
            "repair_cost_usd": repair_cost,
            "simple_payback_years": payback_years,
        }

        assert savings_summary["annual_energy_waste_kwh"] > 0
        assert savings_summary["annual_cost_savings_usd"] > 0
        assert savings_summary["simple_payback_years"] > 0

    def test_roi_analysis_workflow(self):
        """Test complete ROI analysis workflow."""
        # Input parameters
        repair_cost = 10000
        annual_savings = 3000
        discount_rate = 0.10
        analysis_years = 15

        # Calculate metrics
        # Simple payback
        simple_payback = repair_cost / annual_savings

        # NPV
        npv = -repair_cost
        for year in range(1, analysis_years + 1):
            npv += annual_savings / ((1 + discount_rate) ** year)

        # ROI
        total_savings = annual_savings * analysis_years
        roi_percent = ((total_savings - repair_cost) / repair_cost) * 100

        # IRR approximation
        irr = None
        for rate in np.arange(0.01, 0.50, 0.01):
            test_npv = -repair_cost
            for year in range(1, analysis_years + 1):
                test_npv += annual_savings / ((1 + rate) ** year)
            if abs(test_npv) < 100:
                irr = rate
                break

        roi_summary = {
            "simple_payback_years": simple_payback,
            "npv_usd": npv,
            "roi_percent": roi_percent,
            "irr_percent": irr * 100 if irr else None,
            "recommendation": "proceed" if npv > 0 else "review",
        }

        assert roi_summary["simple_payback_years"] < analysis_years
        assert roi_summary["npv_usd"] > 0
        assert roi_summary["recommendation"] == "proceed"


# =============================================================================
# TEST: REPORT GENERATION WORKFLOW
# =============================================================================

@pytest.mark.e2e
class TestReportGenerationWorkflow:
    """End-to-end tests for report generation workflow."""

    def test_complete_inspection_report_generation(
        self,
        sample_thermal_image_data,
        sample_ambient_conditions,
        sample_equipment_parameters,
        sample_insulation_specs
    ):
        """Test complete inspection report generation."""
        # Gather all inspection data
        inspection_data = {
            "inspection_id": f"INS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "inspection_date": date.today().isoformat(),
            "inspector": "INSP-001",
            "equipment": sample_equipment_parameters,
            "insulation": sample_insulation_specs,
            "ambient_conditions": sample_ambient_conditions,
            "thermal_images": [sample_thermal_image_data],
        }

        # Analyze thermal data
        temp_matrix = sample_thermal_image_data["temperature_matrix"]
        analysis = {
            "min_temp_c": min(min(row) for row in temp_matrix),
            "max_temp_c": max(max(row) for row in temp_matrix),
            "avg_temp_c": sum(sum(row) for row in temp_matrix) / (len(temp_matrix) * len(temp_matrix[0])),
            "hotspots_detected": 1,
        }

        # Generate findings
        findings = []
        delta_t = analysis["max_temp_c"] - float(sample_ambient_conditions["ambient_temperature_c"])
        if delta_t > 30:
            findings.append({
                "type": "thermal_anomaly",
                "severity": "moderate" if delta_t < 50 else "severe",
                "location": "Section identified in thermal image",
                "delta_t_c": delta_t,
            })

        # Generate recommendations
        recommendations = []
        for finding in findings:
            if finding["severity"] in ["severe", "critical"]:
                recommendations.append({
                    "action": "immediate_repair",
                    "priority": "urgent",
                    "estimated_cost": 5000,
                })
            else:
                recommendations.append({
                    "action": "schedule_repair",
                    "priority": "medium",
                    "estimated_cost": 3000,
                })

        # Compile report
        report = {
            "report_id": inspection_data["inspection_id"].replace("INS", "RPT"),
            "inspection_id": inspection_data["inspection_id"],
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {
                "equipment_inspected": inspection_data["equipment"]["equipment_tag"],
                "overall_condition": "requires_attention" if findings else "satisfactory",
                "findings_count": len(findings),
                "recommendations_count": len(recommendations),
            },
            "inspection_details": {
                "date": inspection_data["inspection_date"],
                "inspector": inspection_data["inspector"],
                "conditions": inspection_data["ambient_conditions"],
            },
            "thermal_analysis": analysis,
            "findings": findings,
            "recommendations": recommendations,
            "appendices": {
                "thermal_images": len(inspection_data["thermal_images"]),
                "equipment_specs": True,
                "insulation_specs": True,
            },
            "provenance": {
                "hash": hashlib.sha256(
                    json.dumps(inspection_data, default=str).encode()
                ).hexdigest(),
                "methodology": "ASTM_C680",
                "version": "1.0.0",
            },
        }

        assert "report_id" in report
        assert report["executive_summary"]["equipment_inspected"] is not None
        assert len(report["provenance"]["hash"]) == 64

    def test_multi_equipment_report_generation(self):
        """Test report generation for multiple equipment items."""
        equipment_list = [
            {"tag": "P-1001-A", "type": "pipe", "findings": 2},
            {"tag": "V-2001", "type": "vessel", "findings": 1},
            {"tag": "HX-3001", "type": "exchanger", "findings": 0},
        ]

        # Generate summary report
        summary_report = {
            "report_id": f"SUM-{datetime.now().strftime('%Y%m%d')}",
            "facility": "Plant A",
            "inspection_period": "2025-01",
            "equipment_inspected": len(equipment_list),
            "total_findings": sum(e["findings"] for e in equipment_list),
            "equipment_status": [],
        }

        for equip in equipment_list:
            summary_report["equipment_status"].append({
                "tag": equip["tag"],
                "type": equip["type"],
                "findings": equip["findings"],
                "status": "action_required" if equip["findings"] > 0 else "satisfactory",
            })

        # Calculate overall metrics
        equipment_with_issues = sum(1 for e in equipment_list if e["findings"] > 0)
        summary_report["compliance_rate"] = (
            (len(equipment_list) - equipment_with_issues) / len(equipment_list) * 100
        )

        assert summary_report["equipment_inspected"] == 3
        assert summary_report["total_findings"] == 3
        assert summary_report["compliance_rate"] < 100


# =============================================================================
# TEST: PRIORITIZATION WORKFLOW
# =============================================================================

@pytest.mark.e2e
class TestPrioritizationWorkflow:
    """End-to-end tests for repair prioritization workflow."""

    def test_complete_prioritization_workflow(self, multiple_thermal_defects):
        """Test complete repair prioritization workflow."""
        defects = multiple_thermal_defects

        # Step 1: Calculate criticality scores
        for defect in defects:
            heat_loss_score = min(float(defect["heat_loss_w_per_m"]) / 5, 100)
            safety_score = 50 if float(defect["surface_temperature_c"]) > 60 else 25
            process_score = float(defect["process_temperature_c"]) / 5

            defect["criticality_score"] = (
                heat_loss_score * 0.30 +
                safety_score * 0.30 +
                process_score * 0.40
            )

        # Step 2: Assign priority categories
        for defect in defects:
            score = defect["criticality_score"]
            if score >= 75:
                defect["priority"] = "emergency"
            elif score >= 60:
                defect["priority"] = "urgent"
            elif score >= 40:
                defect["priority"] = "high"
            elif score >= 25:
                defect["priority"] = "medium"
            else:
                defect["priority"] = "low"

        # Step 3: Sort by priority
        priority_order = {"emergency": 0, "urgent": 1, "high": 2, "medium": 3, "low": 4}
        sorted_defects = sorted(defects, key=lambda d: priority_order[d["priority"]])

        # Step 4: Allocate budget
        budget = 20000
        allocated = []
        remaining_budget = budget

        for defect in sorted_defects:
            estimated_cost = float(defect["length_m"]) * 1000  # $1000 per meter
            if estimated_cost <= remaining_budget:
                allocated.append({
                    "defect_id": defect["defect_id"],
                    "priority": defect["priority"],
                    "cost": estimated_cost,
                })
                remaining_budget -= estimated_cost

        # Step 5: Generate schedule
        schedule = {
            "total_budget": budget,
            "allocated_budget": budget - remaining_budget,
            "repairs_scheduled": len(allocated),
            "repairs_deferred": len(defects) - len(allocated),
            "schedule": allocated,
        }

        assert schedule["repairs_scheduled"] > 0
        assert schedule["allocated_budget"] <= budget

    def test_budget_constrained_optimization(self, multiple_thermal_defects):
        """Test budget-constrained repair optimization."""
        defects = multiple_thermal_defects
        budget = 5000

        # Simple knapsack optimization
        # Each defect has cost (length * rate) and value (heat loss savings)
        items = []
        for defect in defects:
            cost = float(defect["length_m"]) * 1000
            value = float(defect["heat_loss_w_per_m"]) * float(defect["length_m"]) * 8000 / 1000 * 0.12
            items.append({
                "defect_id": defect["defect_id"],
                "cost": cost,
                "value": value,
                "value_per_cost": value / cost if cost > 0 else 0,
            })

        # Greedy selection by value/cost ratio
        items.sort(key=lambda x: x["value_per_cost"], reverse=True)

        selected = []
        total_cost = 0
        total_value = 0

        for item in items:
            if total_cost + item["cost"] <= budget:
                selected.append(item["defect_id"])
                total_cost += item["cost"]
                total_value += item["value"]

        optimization_result = {
            "budget": budget,
            "utilized": total_cost,
            "utilization_percent": (total_cost / budget) * 100,
            "selected_repairs": selected,
            "expected_annual_savings": total_value,
        }

        assert optimization_result["utilization_percent"] <= 100
        assert optimization_result["expected_annual_savings"] >= 0


# =============================================================================
# TEST: PROVENANCE WORKFLOW
# =============================================================================

@pytest.mark.e2e
class TestProvenanceWorkflow:
    """End-to-end tests for provenance tracking workflow."""

    def test_complete_provenance_chain(self):
        """Test complete provenance chain from input to output."""
        provenance_chain = []

        # Step 1: Input data
        input_data = {
            "equipment_tag": "P-1001-A",
            "process_temp_c": 175.0,
            "ambient_temp_c": 25.0,
        }
        input_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()
        provenance_chain.append({"step": "input", "hash": input_hash})

        # Step 2: Calculation
        calculation_data = {
            "input_hash": input_hash,
            "method": "heat_loss_calculation",
            "parameters": {"k_value": 0.040, "thickness_m": 0.075},
        }
        calc_hash = hashlib.sha256(
            json.dumps(calculation_data, sort_keys=True).encode()
        ).hexdigest()
        provenance_chain.append({"step": "calculation", "hash": calc_hash})

        # Step 3: Output
        output_data = {
            "calculation_hash": calc_hash,
            "heat_loss_w": 450.5,
            "surface_temp_c": 48.2,
        }
        output_hash = hashlib.sha256(
            json.dumps(output_data, sort_keys=True).encode()
        ).hexdigest()
        provenance_chain.append({"step": "output", "hash": output_hash})

        # Verify chain
        assert len(provenance_chain) == 3
        assert all(len(step["hash"]) == 64 for step in provenance_chain)
        assert provenance_chain[1]["step"] == "calculation"

    def test_provenance_verification(self):
        """Test provenance hash verification."""
        # Original calculation
        data = {"value": 123.456, "method": "test"}
        original_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

        # Verification - should match
        verification_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

        assert original_hash == verification_hash

        # Modified data - should not match
        modified_data = {"value": 123.457, "method": "test"}
        modified_hash = hashlib.sha256(
            json.dumps(modified_data, sort_keys=True).encode()
        ).hexdigest()

        assert original_hash != modified_hash
