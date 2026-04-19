# -*- coding: utf-8 -*-
"""
PerformanceReportTemplate - EU Battery Regulation Annex IV Performance & Durability Report

Renders performance and durability parameters for industrial, LMT, and EV
batteries per Annex IV of Regulation (EU) 2023/1542. Covers rated capacity,
voltage characteristics, power capability, cycle life endurance, round-trip
energy efficiency, state of health (SoH), state of charge (SoC), and an
overall durability rating derived from measured parameters against declared
minimum thresholds.

Sections:
    1. Capacity - Rated and remaining capacity, capacity fade
    2. Voltage - Nominal, min, max voltage; voltage stability
    3. Power - Original, max permitted, power fade
    4. Cycle Life - Expected cycles, tested cycles, endurance assessment
    5. Efficiency - Round-trip energy efficiency, initial vs. current
    6. State of Health - SoH determination, degradation tracking
    7. State of Charge - Current SoC, recommended operating range
    8. Durability Rating - Overall durability score and assessment

Author: GreenLang Team
Version: 20.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

_SECTIONS: List[str] = [
    "capacity",
    "voltage",
    "power",
    "cycle_life",
    "efficiency",
    "state_of_health",
    "state_of_charge",
    "durability_rating",
]

# Durability rating thresholds
_DURABILITY_RATINGS: List[Dict[str, Any]] = [
    {"rating": "Excellent", "min_score": 90.0},
    {"rating": "Good", "min_score": 75.0},
    {"rating": "Acceptable", "min_score": 60.0},
    {"rating": "Below Standard", "min_score": 40.0},
    {"rating": "Poor", "min_score": 0.0},
]

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class PerformanceReportTemplate:
    """
    Performance & Durability report template per EU Battery Regulation Annex IV.

    Generates a comprehensive performance assessment covering all electrochemical
    parameters required by Annex IV. Includes capacity retention, voltage stability,
    power capability, cycle life endurance, round-trip energy efficiency, state of
    health tracking, and an overall durability rating.

    Regulatory References:
        - Regulation (EU) 2023/1542, Article 10 + Annex IV
        - IEC 62660-1 (Performance testing for EV batteries)
        - IEC 62620 (Performance of secondary lithium cells for industrial)

    Example:
        >>> tpl = PerformanceReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PerformanceReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "report_id": _new_uuid(),
            "generated_at": self.generated_at.isoformat(),
        }
        for section in _SECTIONS:
            result[section] = self.render_section(section, data)
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_section(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render a single section by name."""
        handler = getattr(self, f"_section_{name}", None)
        if handler is None:
            raise ValueError(f"Unknown section: {name}")
        return handler(data)

    def get_sections(self) -> List[str]:
        """Return list of available section names."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if not data.get("battery_model"):
            errors.append("battery_model is required")
        if not data.get("rated_capacity_kwh"):
            errors.append("rated_capacity_kwh is required for capacity assessment")
        if not data.get("nominal_voltage_v"):
            warnings.append("nominal_voltage_v missing; voltage section limited")
        if not data.get("expected_lifetime_cycles"):
            warnings.append("expected_lifetime_cycles missing; cycle life limited")
        if "round_trip_efficiency_pct" not in data:
            warnings.append("round_trip_efficiency_pct missing")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render performance report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_capacity(data),
            self._md_voltage(data),
            self._md_power(data),
            self._md_cycle_life(data),
            self._md_efficiency(data),
            self._md_state_of_health(data),
            self._md_state_of_charge(data),
            self._md_durability_rating(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render performance report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_capacity(data),
            self._html_voltage(data),
            self._html_power(data),
            self._html_cycle_life(data),
            self._html_efficiency(data),
            self._html_soh_soc(data),
            self._html_durability_rating(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Performance & Durability Report - Annex IV</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render performance report as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "performance_report",
            "regulation_reference": "EU Battery Regulation 2023/1542, Annex IV",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "battery_model": data.get("battery_model", ""),
            "capacity": self._section_capacity(data),
            "voltage": self._section_voltage(data),
            "power": self._section_power(data),
            "cycle_life": self._section_cycle_life(data),
            "efficiency": self._section_efficiency(data),
            "state_of_health": self._section_state_of_health(data),
            "state_of_charge": self._section_state_of_charge(data),
            "durability_rating": self._section_durability_rating(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_capacity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build capacity assessment section."""
        rated = data.get("rated_capacity_kwh", 0.0)
        remaining = data.get("remaining_capacity_kwh", rated)
        fade_pct = data.get("capacity_fade_pct", 0.0)
        if rated > 0 and remaining > 0 and fade_pct == 0.0:
            fade_pct = round((1.0 - remaining / rated) * 100, 2)
        retention_pct = round(100.0 - fade_pct, 2)

        return {
            "title": "Capacity Assessment",
            "rated_capacity_kwh": round(rated, 3),
            "rated_capacity_ah": data.get("rated_capacity_ah", 0.0),
            "remaining_capacity_kwh": round(remaining, 3),
            "capacity_fade_pct": round(fade_pct, 2),
            "capacity_retention_pct": retention_pct,
            "min_capacity_threshold_pct": data.get("min_capacity_threshold_pct", 80.0),
            "above_threshold": retention_pct >= data.get("min_capacity_threshold_pct", 80.0),
            "measurement_date": data.get("capacity_measurement_date", ""),
            "measurement_conditions": data.get(
                "capacity_measurement_conditions",
                "25C, 0.2C discharge rate per IEC 62660-1",
            ),
        }

    def _section_voltage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build voltage characteristics section."""
        nominal = data.get("nominal_voltage_v", 0.0)
        min_v = data.get("minimum_voltage_v", 0.0)
        max_v = data.get("maximum_voltage_v", 0.0)
        voltage_range = round(max_v - min_v, 2) if max_v > 0 and min_v > 0 else 0.0
        return {
            "title": "Voltage Characteristics",
            "nominal_voltage_v": round(nominal, 2),
            "minimum_voltage_v": round(min_v, 2),
            "maximum_voltage_v": round(max_v, 2),
            "voltage_range_v": voltage_range,
            "open_circuit_voltage_v": data.get("open_circuit_voltage_v", 0.0),
            "cell_count_series": data.get("cell_count_series", 0),
            "cell_count_parallel": data.get("cell_count_parallel", 0),
            "cell_nominal_voltage_v": data.get("cell_nominal_voltage_v", 0.0),
            "voltage_stability_assessment": self._assess_voltage_stability(data),
        }

    def _section_power(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build power capability section."""
        original_power = data.get("original_power_w", 0.0)
        current_power = data.get("current_power_w", original_power)
        max_permitted = data.get("max_permitted_power_w", 0.0)
        power_fade = data.get("power_fade_pct", 0.0)
        if original_power > 0 and current_power > 0 and power_fade == 0.0:
            power_fade = round((1.0 - current_power / original_power) * 100, 2)

        return {
            "title": "Power Capability",
            "original_power_w": round(original_power, 1),
            "current_power_w": round(current_power, 1),
            "max_permitted_power_w": round(max_permitted, 1),
            "power_fade_pct": round(power_fade, 2),
            "power_retention_pct": round(100.0 - power_fade, 2),
            "peak_power_30s_w": data.get("peak_power_30s_w", 0.0),
            "continuous_power_w": data.get("continuous_power_w", 0.0),
            "specific_power_w_per_kg": data.get("specific_power_w_per_kg", 0.0),
            "power_density_w_per_l": data.get("power_density_w_per_l", 0.0),
        }

    def _section_cycle_life(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build cycle life endurance section."""
        expected = data.get("expected_lifetime_cycles", 0)
        completed = data.get("completed_cycles", 0)
        remaining = max(0, expected - completed) if expected > 0 else 0
        utilization_pct = (
            round(completed / expected * 100, 1) if expected > 0 else 0.0
        )
        return {
            "title": "Cycle Life Endurance",
            "expected_lifetime_cycles": expected,
            "completed_cycles": completed,
            "remaining_cycles": remaining,
            "cycle_utilization_pct": utilization_pct,
            "reference_test_standard": data.get(
                "cycle_life_reference_test", "IEC 62660-1"
            ),
            "test_c_rate": data.get("c_rate", "1C charge / 1C discharge"),
            "test_temperature_c": data.get("test_temperature_c", 25),
            "depth_of_discharge_pct": data.get("depth_of_discharge_pct", 80.0),
            "energy_throughput_kwh": data.get("energy_throughput_kwh", 0.0),
            "expected_energy_throughput_kwh": data.get(
                "expected_energy_throughput_kwh", 0.0
            ),
            "endurance_status": self._endurance_status(utilization_pct),
        }

    def _section_efficiency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build round-trip energy efficiency section."""
        current_eff = data.get("round_trip_efficiency_pct", 0.0)
        initial_eff = data.get("initial_round_trip_efficiency_pct", current_eff)
        degradation = round(initial_eff - current_eff, 2) if initial_eff > 0 else 0.0
        return {
            "title": "Round-Trip Energy Efficiency",
            "initial_efficiency_pct": round(initial_eff, 2),
            "current_efficiency_pct": round(current_eff, 2),
            "efficiency_degradation_pct": degradation,
            "min_efficiency_threshold_pct": data.get("min_efficiency_threshold_pct", 0.0),
            "above_threshold": (
                current_eff >= data.get("min_efficiency_threshold_pct", 0.0)
                if data.get("min_efficiency_threshold_pct", 0.0) > 0
                else True
            ),
            "measurement_conditions": data.get(
                "efficiency_measurement_conditions",
                "25C, rated power, full cycle",
            ),
            "coulombic_efficiency_pct": data.get("coulombic_efficiency_pct", 0.0),
            "self_discharge_rate_pct_per_month": data.get(
                "self_discharge_rate_pct_per_month", 0.0
            ),
        }

    def _section_state_of_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build state of health section."""
        soh = data.get("state_of_health_pct", 100.0)
        soh_history = data.get("soh_history", [])
        return {
            "title": "State of Health (SoH)",
            "current_soh_pct": round(soh, 2),
            "soh_determination_method": data.get(
                "soh_method", "Capacity retention vs. rated capacity"
            ),
            "soh_threshold_eol_pct": data.get("soh_threshold_eol_pct", 80.0),
            "remaining_useful_life_pct": round(
                max(0.0, soh - data.get("soh_threshold_eol_pct", 80.0))
                / (100.0 - data.get("soh_threshold_eol_pct", 80.0)) * 100, 1
            ) if soh > data.get("soh_threshold_eol_pct", 80.0) else 0.0,
            "degradation_rate_pct_per_year": data.get(
                "degradation_rate_pct_per_year", 0.0
            ),
            "estimated_remaining_years": self._estimate_remaining_years(data),
            "soh_history": [
                {
                    "date": entry.get("date", ""),
                    "soh_pct": entry.get("soh_pct", 0.0),
                    "cycles_at_measurement": entry.get("cycles", 0),
                }
                for entry in soh_history
            ],
            "warranty_soh_threshold_pct": data.get("warranty_soh_threshold_pct", 70.0),
            "above_warranty_threshold": soh >= data.get("warranty_soh_threshold_pct", 70.0),
        }

    def _section_state_of_charge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build state of charge section."""
        soc = data.get("state_of_charge_pct", 0.0)
        return {
            "title": "State of Charge (SoC)",
            "current_soc_pct": round(soc, 2),
            "recommended_min_soc_pct": data.get("recommended_min_soc_pct", 10.0),
            "recommended_max_soc_pct": data.get("recommended_max_soc_pct", 90.0),
            "within_recommended_range": (
                data.get("recommended_min_soc_pct", 10.0)
                <= soc
                <= data.get("recommended_max_soc_pct", 90.0)
            ),
            "soc_accuracy_pct": data.get("soc_accuracy_pct", 5.0),
            "soc_measurement_method": data.get(
                "soc_measurement_method", "Coulomb counting with OCV correction"
            ),
            "last_full_charge_date": data.get("last_full_charge_date", ""),
            "usable_capacity_at_soc_kwh": round(
                data.get("rated_capacity_kwh", 0.0) * soc / 100.0, 3
            ),
        }

    def _section_durability_rating(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build overall durability rating section."""
        scores = self._calculate_durability_scores(data)
        overall_score = scores["overall_score"]
        rating = self._determine_rating(overall_score)

        return {
            "title": "Overall Durability Rating",
            "overall_score": round(overall_score, 1),
            "rating": rating,
            "component_scores": {
                "capacity_retention_score": scores["capacity_score"],
                "power_retention_score": scores["power_score"],
                "cycle_life_score": scores["cycle_score"],
                "efficiency_score": scores["efficiency_score"],
                "soh_score": scores["soh_score"],
            },
            "rating_scale": [
                {"rating": r["rating"], "min_score": r["min_score"]}
                for r in _DURABILITY_RATINGS
            ],
            "assessment_date": data.get(
                "assessment_date", utcnow().strftime("%Y-%m-%d")
            ),
            "assessor": data.get("assessor", ""),
            "next_assessment_due": data.get("next_assessment_due", ""),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Performance & Durability Report\n"
            f"## EU Battery Regulation (EU) 2023/1542 - Annex IV\n\n"
            f"**Manufacturer:** {data.get('entity_name', '')}  \n"
            f"**Battery Model:** {data.get('battery_model', '')}  \n"
            f"**Battery Type:** {data.get('battery_type', 'ev_battery')}  \n"
            f"**Generated:** {ts}"
        )

    def _md_capacity(self, data: Dict[str, Any]) -> str:
        """Render capacity section as markdown."""
        sec = self._section_capacity(data)
        threshold_status = "PASS" if sec["above_threshold"] else "BELOW THRESHOLD"
        return (
            f"## {sec['title']}\n\n"
            f"| Parameter | Value |\n|-----------|------:|\n"
            f"| Rated Capacity | {sec['rated_capacity_kwh']:.3f} kWh |\n"
            f"| Remaining Capacity | {sec['remaining_capacity_kwh']:.3f} kWh |\n"
            f"| Capacity Fade | {sec['capacity_fade_pct']:.2f}% |\n"
            f"| Capacity Retention | {sec['capacity_retention_pct']:.2f}% |\n"
            f"| Min Threshold | {sec['min_capacity_threshold_pct']:.1f}% |\n"
            f"| Status | **{threshold_status}** |\n\n"
            f"*Conditions: {sec['measurement_conditions']}*"
        )

    def _md_voltage(self, data: Dict[str, Any]) -> str:
        """Render voltage section as markdown."""
        sec = self._section_voltage(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Parameter | Value |\n|-----------|------:|\n"
            f"| Nominal Voltage | {sec['nominal_voltage_v']:.2f} V |\n"
            f"| Minimum Voltage | {sec['minimum_voltage_v']:.2f} V |\n"
            f"| Maximum Voltage | {sec['maximum_voltage_v']:.2f} V |\n"
            f"| Voltage Range | {sec['voltage_range_v']:.2f} V |\n"
            f"| Cells (Series x Parallel) | {sec['cell_count_series']}s"
            f"{sec['cell_count_parallel']}p |\n"
            f"| Stability | {sec['voltage_stability_assessment']} |"
        )

    def _md_power(self, data: Dict[str, Any]) -> str:
        """Render power section as markdown."""
        sec = self._section_power(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Parameter | Value |\n|-----------|------:|\n"
            f"| Original Power | {sec['original_power_w']:.1f} W |\n"
            f"| Current Power | {sec['current_power_w']:.1f} W |\n"
            f"| Max Permitted | {sec['max_permitted_power_w']:.1f} W |\n"
            f"| Power Fade | {sec['power_fade_pct']:.2f}% |\n"
            f"| Power Retention | {sec['power_retention_pct']:.2f}% |\n"
            f"| Peak Power (30s) | {sec['peak_power_30s_w']:.1f} W |\n"
            f"| Specific Power | {sec['specific_power_w_per_kg']:.1f} W/kg |"
        )

    def _md_cycle_life(self, data: Dict[str, Any]) -> str:
        """Render cycle life section as markdown."""
        sec = self._section_cycle_life(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Parameter | Value |\n|-----------|------:|\n"
            f"| Expected Lifetime | {sec['expected_lifetime_cycles']} cycles |\n"
            f"| Completed Cycles | {sec['completed_cycles']} |\n"
            f"| Remaining Cycles | {sec['remaining_cycles']} |\n"
            f"| Utilization | {sec['cycle_utilization_pct']:.1f}% |\n"
            f"| Energy Throughput | {sec['energy_throughput_kwh']:.1f} kWh |\n"
            f"| DoD | {sec['depth_of_discharge_pct']:.1f}% |\n"
            f"| Endurance | **{sec['endurance_status']}** |\n\n"
            f"*Test: {sec['reference_test_standard']}, "
            f"{sec['test_c_rate']}, {sec['test_temperature_c']}C*"
        )

    def _md_efficiency(self, data: Dict[str, Any]) -> str:
        """Render efficiency section as markdown."""
        sec = self._section_efficiency(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Parameter | Value |\n|-----------|------:|\n"
            f"| Initial Efficiency | {sec['initial_efficiency_pct']:.2f}% |\n"
            f"| Current Efficiency | {sec['current_efficiency_pct']:.2f}% |\n"
            f"| Degradation | {sec['efficiency_degradation_pct']:.2f}% |\n"
            f"| Coulombic Efficiency | {sec['coulombic_efficiency_pct']:.2f}% |\n"
            f"| Self-Discharge | {sec['self_discharge_rate_pct_per_month']:.2f}%/month |"
        )

    def _md_state_of_health(self, data: Dict[str, Any]) -> str:
        """Render state of health section as markdown."""
        sec = self._section_state_of_health(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Current SoH:** {sec['current_soh_pct']:.2f}%  \n"
            f"**EoL Threshold:** {sec['soh_threshold_eol_pct']:.1f}%  \n"
            f"**Degradation Rate:** {sec['degradation_rate_pct_per_year']:.2f}%/year  \n"
            f"**Est. Remaining Years:** {sec['estimated_remaining_years']:.1f}  \n"
            f"**Method:** {sec['soh_determination_method']}",
        ]
        if sec["soh_history"]:
            lines.append("\n### SoH History\n")
            lines.append("| Date | SoH (%) | Cycles |")
            lines.append("|------|--------:|-------:|")
            for entry in sec["soh_history"]:
                lines.append(
                    f"| {entry['date']} | {entry['soh_pct']:.1f}% | "
                    f"{entry['cycles_at_measurement']} |"
                )
        return "\n".join(lines)

    def _md_state_of_charge(self, data: Dict[str, Any]) -> str:
        """Render state of charge section as markdown."""
        sec = self._section_state_of_charge(data)
        range_status = "IN RANGE" if sec["within_recommended_range"] else "OUT OF RANGE"
        return (
            f"## {sec['title']}\n\n"
            f"| Parameter | Value |\n|-----------|------:|\n"
            f"| Current SoC | {sec['current_soc_pct']:.2f}% |\n"
            f"| Recommended Range | {sec['recommended_min_soc_pct']:.0f}% - "
            f"{sec['recommended_max_soc_pct']:.0f}% |\n"
            f"| Range Status | **{range_status}** |\n"
            f"| SoC Accuracy | +/- {sec['soc_accuracy_pct']:.1f}% |\n"
            f"| Usable Capacity | {sec['usable_capacity_at_soc_kwh']:.3f} kWh |"
        )

    def _md_durability_rating(self, data: Dict[str, Any]) -> str:
        """Render durability rating section as markdown."""
        sec = self._section_durability_rating(data)
        scores = sec["component_scores"]
        lines = [
            f"## {sec['title']}\n",
            f"### Overall: {sec['overall_score']:.1f}/100 - {sec['rating']}\n",
            "| Component | Score |",
            "|-----------|------:|",
            f"| Capacity Retention | {scores['capacity_retention_score']:.1f} |",
            f"| Power Retention | {scores['power_retention_score']:.1f} |",
            f"| Cycle Life | {scores['cycle_life_score']:.1f} |",
            f"| Efficiency | {scores['efficiency_score']:.1f} |",
            f"| State of Health | {scores['soh_score']:.1f} |",
            f"\n**Assessment Date:** {sec['assessment_date']}",
        ]
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n*Report generated by PACK-020 Battery Passport Prep Pack on {ts}*\n"
            f"*Regulation: EU Battery Regulation (EU) 2023/1542, Annex IV*"
        )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1000px;margin:auto}"
            "h1{color:#0d47a1;border-bottom:2px solid #0d47a1;padding-bottom:.3em}"
            "h2{color:#1565c0;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e3f2fd}"
            ".pass{color:#2e7d32;font-weight:bold}"
            ".fail{color:#c62828;font-weight:bold}"
            ".rating{font-size:1.5em;font-weight:bold;padding:8px 16px;"
            "border-radius:4px;display:inline-block}"
            ".rating-excellent{background:#c8e6c9;color:#1b5e20}"
            ".rating-good{background:#dcedc8;color:#33691e}"
            ".rating-acceptable{background:#fff9c4;color:#f57f17}"
            ".rating-below{background:#ffe0b2;color:#e65100}"
            ".rating-poor{background:#ffcdd2;color:#b71c1c}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Performance & Durability Report</h1>\n"
            f"<p>EU Battery Regulation (EU) 2023/1542 - Annex IV</p>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('battery_model', '')}</p>"
        )

    def _html_capacity(self, data: Dict[str, Any]) -> str:
        """Render capacity HTML."""
        sec = self._section_capacity(data)
        cls = "pass" if sec["above_threshold"] else "fail"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Parameter</th><th>Value</th></tr>"
            f"<tr><td>Rated Capacity</td><td>{sec['rated_capacity_kwh']:.3f} kWh</td></tr>"
            f"<tr><td>Remaining</td><td>{sec['remaining_capacity_kwh']:.3f} kWh</td></tr>"
            f"<tr><td>Retention</td>"
            f"<td class='{cls}'>{sec['capacity_retention_pct']:.2f}%</td></tr></table>"
        )

    def _html_voltage(self, data: Dict[str, Any]) -> str:
        """Render voltage HTML."""
        sec = self._section_voltage(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Parameter</th><th>Value</th></tr>"
            f"<tr><td>Nominal</td><td>{sec['nominal_voltage_v']:.2f} V</td></tr>"
            f"<tr><td>Range</td><td>{sec['minimum_voltage_v']:.2f} - "
            f"{sec['maximum_voltage_v']:.2f} V</td></tr></table>"
        )

    def _html_power(self, data: Dict[str, Any]) -> str:
        """Render power HTML."""
        sec = self._section_power(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Parameter</th><th>Value</th></tr>"
            f"<tr><td>Original Power</td><td>{sec['original_power_w']:.1f} W</td></tr>"
            f"<tr><td>Current Power</td><td>{sec['current_power_w']:.1f} W</td></tr>"
            f"<tr><td>Power Fade</td><td>{sec['power_fade_pct']:.2f}%</td></tr></table>"
        )

    def _html_cycle_life(self, data: Dict[str, Any]) -> str:
        """Render cycle life HTML."""
        sec = self._section_cycle_life(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Parameter</th><th>Value</th></tr>"
            f"<tr><td>Expected</td><td>{sec['expected_lifetime_cycles']} cycles</td></tr>"
            f"<tr><td>Completed</td><td>{sec['completed_cycles']}</td></tr>"
            f"<tr><td>Remaining</td><td>{sec['remaining_cycles']}</td></tr>"
            f"<tr><td>Utilization</td><td>{sec['cycle_utilization_pct']:.1f}%</td></tr>"
            f"</table>"
        )

    def _html_efficiency(self, data: Dict[str, Any]) -> str:
        """Render efficiency HTML."""
        sec = self._section_efficiency(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Current: <strong>{sec['current_efficiency_pct']:.2f}%</strong> "
            f"(initial: {sec['initial_efficiency_pct']:.2f}%)</p>"
        )

    def _html_soh_soc(self, data: Dict[str, Any]) -> str:
        """Render SoH and SoC HTML."""
        soh = self._section_state_of_health(data)
        soc = self._section_state_of_charge(data)
        return (
            f"<h2>State of Health & Charge</h2>\n"
            f"<table><tr><th>Parameter</th><th>Value</th></tr>"
            f"<tr><td>SoH</td><td>{soh['current_soh_pct']:.2f}%</td></tr>"
            f"<tr><td>SoC</td><td>{soc['current_soc_pct']:.2f}%</td></tr></table>"
        )

    def _html_durability_rating(self, data: Dict[str, Any]) -> str:
        """Render durability rating HTML."""
        sec = self._section_durability_rating(data)
        rating_cls = sec["rating"].lower().replace(" ", "-")
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='rating rating-{rating_cls}'>"
            f"{sec['overall_score']:.1f}/100 - {sec['rating']}</p>"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assess_voltage_stability(self, data: Dict[str, Any]) -> str:
        """Assess voltage stability from operating parameters."""
        nominal = data.get("nominal_voltage_v", 0.0)
        min_v = data.get("minimum_voltage_v", 0.0)
        max_v = data.get("maximum_voltage_v", 0.0)
        if nominal <= 0 or min_v <= 0 or max_v <= 0:
            return "Insufficient data"
        spread_pct = (max_v - min_v) / nominal * 100
        if spread_pct <= 20.0:
            return "Stable"
        elif spread_pct <= 35.0:
            return "Moderate"
        return "Wide range"

    def _endurance_status(self, utilization_pct: float) -> str:
        """Determine endurance status from cycle utilization."""
        if utilization_pct < 25.0:
            return "Early life"
        elif utilization_pct < 50.0:
            return "Mid-life"
        elif utilization_pct < 75.0:
            return "Mature"
        elif utilization_pct < 100.0:
            return "Late life"
        return "Beyond expected life"

    def _estimate_remaining_years(self, data: Dict[str, Any]) -> float:
        """Estimate remaining useful life in years."""
        soh = data.get("state_of_health_pct", 100.0)
        threshold = data.get("soh_threshold_eol_pct", 80.0)
        rate = data.get("degradation_rate_pct_per_year", 0.0)
        if rate <= 0 or soh <= threshold:
            return 0.0
        return round((soh - threshold) / rate, 1)

    def _calculate_durability_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate component durability scores (0-100 scale)."""
        # Capacity: 100 = no fade, 0 = 20%+ fade
        cap_fade = data.get("capacity_fade_pct", 0.0)
        cap_score = max(0.0, min(100.0, 100.0 - cap_fade * 5.0))

        # Power: 100 = no fade, 0 = 20%+ fade
        pwr_fade = data.get("power_fade_pct", 0.0)
        pwr_score = max(0.0, min(100.0, 100.0 - pwr_fade * 5.0))

        # Cycle life: based on remaining percentage
        expected = data.get("expected_lifetime_cycles", 0)
        completed = data.get("completed_cycles", 0)
        cycle_score = (
            max(0.0, min(100.0, (1.0 - completed / expected) * 100.0))
            if expected > 0 else 100.0
        )

        # Efficiency: 100 if >= 95%, scales down
        eff = data.get("round_trip_efficiency_pct", 95.0)
        eff_score = max(0.0, min(100.0, eff / 0.95))

        # SoH: direct mapping
        soh = data.get("state_of_health_pct", 100.0)
        soh_score = max(0.0, min(100.0, soh))

        # Weighted average
        weights = {"capacity": 0.25, "power": 0.15, "cycle": 0.25,
                    "efficiency": 0.15, "soh": 0.20}
        overall = (
            cap_score * weights["capacity"]
            + pwr_score * weights["power"]
            + cycle_score * weights["cycle"]
            + eff_score * weights["efficiency"]
            + soh_score * weights["soh"]
        )

        return {
            "capacity_score": round(cap_score, 1),
            "power_score": round(pwr_score, 1),
            "cycle_score": round(cycle_score, 1),
            "efficiency_score": round(eff_score, 1),
            "soh_score": round(soh_score, 1),
            "overall_score": round(overall, 1),
        }

    def _determine_rating(self, score: float) -> str:
        """Determine durability rating from overall score."""
        for r in _DURABILITY_RATINGS:
            if score >= r["min_score"]:
                return r["rating"]
        return "Poor"
