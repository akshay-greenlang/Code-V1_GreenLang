# -*- coding: utf-8 -*-
"""
BatteryPassportReportTemplate - EU Battery Regulation Annex XIII Full Battery Passport

Renders the comprehensive battery passport as required by Annex XIII of
Regulation (EU) 2023/1542. The battery passport is a digital record
associated with each industrial, LMT, and EV battery placed on the EU
market from 18 February 2027. It contains general battery information,
carbon footprint data, supply chain due diligence results, material
composition, performance and durability characteristics, end-of-life
information, and a QR code data payload for machine-readable access.

Sections:
    1. General Information - Manufacturer, model, chemistry, capacity
    2. Carbon Footprint - Lifecycle CF and performance class
    3. Supply Chain Due Diligence - Responsible sourcing compliance
    4. Material Composition - Hazardous substances, critical raw materials
    5. Performance & Durability - Rated capacity, cycle life, SoH
    6. End-of-Life Information - Collection, recycling, second life
    7. QR Code Data Payload - Machine-readable identifier and access URL

Author: GreenLang Team
Version: 20.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

_SECTIONS: List[str] = [
    "general_information",
    "carbon_footprint",
    "supply_chain_due_diligence",
    "material_composition",
    "performance_durability",
    "end_of_life",
    "qr_code_data",
]

# Battery chemistries commonly reported
_BATTERY_CHEMISTRIES: List[str] = [
    "NMC (Nickel Manganese Cobalt)",
    "NCA (Nickel Cobalt Aluminium)",
    "LFP (Lithium Iron Phosphate)",
    "LTO (Lithium Titanate)",
    "Lead-acid",
    "NiMH (Nickel Metal Hydride)",
    "Solid-state",
    "Other",
]


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class BatteryPassportReportTemplate:
    """
    Full Battery Passport report template per EU Battery Regulation Annex XIII.

    Generates the complete digital battery passport containing all mandatory
    data points required by Annex XIII of Regulation (EU) 2023/1542 for
    industrial, LMT, and EV batteries placed on the EU market from
    18 February 2027.

    Regulatory References:
        - Regulation (EU) 2023/1542, Articles 77-78
        - Regulation (EU) 2023/1542, Annex XIII
        - Commission Implementing Regulation on battery passport technical design

    Example:
        >>> tpl = BatteryPassportReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BatteryPassportReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = _utcnow()
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
        required = ["entity_name", "battery_model", "battery_unique_id"]
        for field in required:
            if not data.get(field):
                errors.append(f"{field} is required")
        if not data.get("battery_chemistry"):
            warnings.append("battery_chemistry not specified")
        if not data.get("carbon_footprint"):
            warnings.append("carbon_footprint data missing; section will be limited")
        if not data.get("materials"):
            warnings.append("materials data missing; composition section limited")
        if not data.get("performance"):
            warnings.append("performance data missing; section will be limited")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render battery passport report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_general_information(data),
            self._md_carbon_footprint(data),
            self._md_supply_chain_dd(data),
            self._md_material_composition(data),
            self._md_performance_durability(data),
            self._md_end_of_life(data),
            self._md_qr_code_data(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render battery passport report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_general_information(data),
            self._html_carbon_footprint(data),
            self._html_supply_chain_dd(data),
            self._html_material_composition(data),
            self._html_performance_durability(data),
            self._html_end_of_life(data),
            self._html_qr_code_data(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Battery Passport - Annex XIII</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render battery passport report as JSON."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "battery_passport_report",
            "regulation_reference": "EU Battery Regulation 2023/1542, Annex XIII",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "battery_unique_id": data.get("battery_unique_id", ""),
            "general_information": self._section_general_information(data),
            "carbon_footprint": self._section_carbon_footprint(data),
            "supply_chain_due_diligence": self._section_supply_chain_due_diligence(data),
            "material_composition": self._section_material_composition(data),
            "performance_durability": self._section_performance_durability(data),
            "end_of_life": self._section_end_of_life(data),
            "qr_code_data": self._section_qr_code_data(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_general_information(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build general information section per Annex XIII Part A."""
        return {
            "title": "General Battery Information",
            "annex_reference": "Annex XIII, Part A",
            "manufacturer_name": data.get("entity_name", ""),
            "manufacturer_address": data.get("manufacturer_address", ""),
            "manufacturer_id": data.get("manufacturer_id", ""),
            "battery_unique_id": data.get("battery_unique_id", ""),
            "battery_model": data.get("battery_model", ""),
            "battery_type": data.get("battery_type", "ev_battery"),
            "battery_chemistry": data.get("battery_chemistry", ""),
            "nominal_voltage_v": data.get("nominal_voltage_v", 0.0),
            "rated_capacity_kwh": data.get("rated_capacity_kwh", 0.0),
            "rated_capacity_ah": data.get("rated_capacity_ah", 0.0),
            "weight_kg": data.get("weight_kg", 0.0),
            "manufacturing_date": data.get("manufacturing_date", ""),
            "manufacturing_location": data.get("manufacturing_location", ""),
            "eu_declaration_of_conformity_id": data.get("eu_doc_id", ""),
            "notified_body_id": data.get("notified_body_id", ""),
            "warranty_period_years": data.get("warranty_period_years", 0),
            "expected_lifetime_years": data.get("expected_lifetime_years", 0),
            "expected_lifetime_cycles": data.get("expected_lifetime_cycles", 0),
        }

    def _section_carbon_footprint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build carbon footprint section per Annex XIII Part B."""
        cf_data = data.get("carbon_footprint", {})
        total_cf = cf_data.get("total_kgco2e_per_kwh", 0.0)
        return {
            "title": "Carbon Footprint Information",
            "annex_reference": "Annex XIII, Part B",
            "total_carbon_footprint_kgco2e_per_kwh": round(total_cf, 2),
            "performance_class": cf_data.get("performance_class", ""),
            "carbon_footprint_share_by_stage": {
                "raw_material_acquisition_pct": cf_data.get("raw_material_pct", 0.0),
                "main_production_pct": cf_data.get("production_pct", 0.0),
                "distribution_pct": cf_data.get("distribution_pct", 0.0),
                "end_of_life_pct": cf_data.get("end_of_life_pct", 0.0),
            },
            "web_link_to_cf_study": cf_data.get("cf_study_url", ""),
            "declaration_number": cf_data.get("declaration_number", ""),
            "methodology_reference": cf_data.get(
                "methodology", "Commission Delegated Regulation (EU) 2023/1791"
            ),
        }

    def _section_supply_chain_due_diligence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build supply chain due diligence section per Annex XIII Part C."""
        dd_data = data.get("due_diligence", {})
        suppliers = dd_data.get("suppliers", [])
        assessed_count = sum(1 for s in suppliers if s.get("assessed", False))
        return {
            "title": "Supply Chain Due Diligence",
            "annex_reference": "Annex XIII, Part C",
            "dd_policy_available": dd_data.get("policy_available", False),
            "dd_policy_url": dd_data.get("policy_url", ""),
            "oecd_guidance_compliant": dd_data.get("oecd_compliant", False),
            "total_suppliers": len(suppliers),
            "assessed_suppliers": assessed_count,
            "assessment_coverage_pct": (
                round(assessed_count / len(suppliers) * 100, 1)
                if suppliers else 0.0
            ),
            "third_party_audit_completed": dd_data.get("third_party_audit", False),
            "grievance_mechanism_established": dd_data.get("grievance_mechanism", False),
            "risk_areas_identified": dd_data.get("risk_areas", []),
            "raw_material_origins": dd_data.get("origins", []),
            "conflict_minerals_assessment": dd_data.get("conflict_assessment", ""),
            "child_labour_risk_assessment": dd_data.get("child_labour_assessment", ""),
        }

    def _section_material_composition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build material composition section per Annex XIII Part D."""
        materials = data.get("materials", [])
        hazardous = data.get("hazardous_substances", [])
        critical_raw = data.get("critical_raw_materials", [])

        material_details: List[Dict[str, Any]] = []
        for mat in materials:
            material_details.append({
                "name": mat.get("name", ""),
                "mass_kg": round(mat.get("total_mass_kg", 0.0), 3),
                "mass_percentage": mat.get("mass_percentage", 0.0),
                "location_in_battery": mat.get("location", ""),
                "is_hazardous": mat.get("is_hazardous", False),
                "is_critical_raw_material": mat.get("is_critical", False),
                "recycled_content_pct": mat.get("recycled_content_pct", 0.0),
                "cas_number": mat.get("cas_number", ""),
            })

        return {
            "title": "Material Composition",
            "annex_reference": "Annex XIII, Part D",
            "total_material_count": len(material_details),
            "materials": material_details,
            "hazardous_substances": [
                {
                    "name": h.get("name", ""),
                    "cas_number": h.get("cas_number", ""),
                    "concentration_pct": h.get("concentration_pct", 0.0),
                    "location": h.get("location", ""),
                    "reach_registered": h.get("reach_registered", False),
                }
                for h in hazardous
            ],
            "critical_raw_materials": [
                {
                    "name": crm.get("name", ""),
                    "mass_kg": round(crm.get("mass_kg", 0.0), 3),
                    "source_country": crm.get("source_country", ""),
                    "on_eu_crm_list": crm.get("on_eu_list", True),
                }
                for crm in critical_raw
            ],
            "hazardous_count": len(hazardous),
            "critical_raw_material_count": len(critical_raw),
        }

    def _section_performance_durability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build performance and durability section per Annex XIII Part E."""
        perf = data.get("performance", {})
        return {
            "title": "Performance & Durability",
            "annex_reference": "Annex XIII, Part E",
            "rated_capacity_kwh": perf.get("rated_capacity_kwh", 0.0),
            "rated_capacity_ah": perf.get("rated_capacity_ah", 0.0),
            "nominal_voltage_v": perf.get("nominal_voltage_v", 0.0),
            "minimum_voltage_v": perf.get("minimum_voltage_v", 0.0),
            "maximum_voltage_v": perf.get("maximum_voltage_v", 0.0),
            "original_power_w": perf.get("original_power_w", 0.0),
            "max_permitted_power_w": perf.get("max_permitted_power_w", 0.0),
            "round_trip_efficiency_pct": perf.get("round_trip_efficiency_pct", 0.0),
            "initial_round_trip_efficiency_pct": perf.get(
                "initial_round_trip_efficiency_pct", 0.0
            ),
            "expected_lifetime_cycles": perf.get("expected_lifetime_cycles", 0),
            "cycle_life_reference_test": perf.get("cycle_life_reference_test", ""),
            "c_rate": perf.get("c_rate", ""),
            "energy_throughput_kwh": perf.get("energy_throughput_kwh", 0.0),
            "capacity_fade_pct": perf.get("capacity_fade_pct", 0.0),
            "power_fade_pct": perf.get("power_fade_pct", 0.0),
            "internal_resistance_ohm": perf.get("internal_resistance_ohm", 0.0),
            "internal_resistance_increase_pct": perf.get(
                "internal_resistance_increase_pct", 0.0
            ),
            "temperature_range_min_c": perf.get("temperature_range_min_c", 0),
            "temperature_range_max_c": perf.get("temperature_range_max_c", 0),
            "state_of_health_pct": perf.get("state_of_health_pct", 100.0),
            "state_of_charge_pct": perf.get("state_of_charge_pct", 0.0),
        }

    def _section_end_of_life(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build end-of-life information section per Annex XIII Part F."""
        eol = data.get("end_of_life", {})
        return {
            "title": "End-of-Life Information",
            "annex_reference": "Annex XIII, Part F",
            "collection_scheme_info": eol.get("collection_scheme", ""),
            "return_point_info": eol.get("return_points", ""),
            "dismantling_instructions_url": eol.get("dismantling_url", ""),
            "safety_instructions_url": eol.get("safety_url", ""),
            "recycler_contact": eol.get("recycler_contact", ""),
            "second_life_applicable": eol.get("second_life_applicable", False),
            "second_life_requirements": eol.get("second_life_requirements", ""),
            "repurposing_info": eol.get("repurposing_info", ""),
            "waste_prevention_measures": eol.get("waste_prevention", []),
            "recyclability_assessment": eol.get("recyclability_assessment", ""),
            "recycling_efficiency_target_pct": eol.get(
                "recycling_efficiency_target_pct", 0.0
            ),
            "material_recovery_targets": eol.get("material_recovery_targets", {}),
        }

    def _section_qr_code_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build QR code data payload section per Art 77(4)."""
        battery_uid = data.get("battery_unique_id", "")
        passport_url = data.get(
            "passport_url",
            f"https://batterypassport.eu/passport/{battery_uid}",
        )
        qr_payload = {
            "battery_unique_id": battery_uid,
            "passport_url": passport_url,
            "manufacturer_id": data.get("manufacturer_id", ""),
            "battery_model": data.get("battery_model", ""),
        }
        return {
            "title": "QR Code Data Payload",
            "article_reference": "Art 77(4)",
            "battery_unique_id": battery_uid,
            "passport_url": passport_url,
            "qr_payload_json": json.dumps(qr_payload, sort_keys=True),
            "qr_payload_hash": _compute_hash(qr_payload),
            "qr_must_be_on_battery": True,
            "qr_must_link_to_passport": True,
            "access_rights_note": (
                "Public access to Annex XIII Part A data; "
                "restricted access to Parts B-F per Art 77(5)"
            ),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Battery Passport\n"
            f"## EU Battery Regulation (EU) 2023/1542 - Annex XIII\n\n"
            f"**Battery ID:** {data.get('battery_unique_id', '')}  \n"
            f"**Manufacturer:** {data.get('entity_name', '')}  \n"
            f"**Model:** {data.get('battery_model', '')}  \n"
            f"**Chemistry:** {data.get('battery_chemistry', '')}  \n"
            f"**Generated:** {ts}"
        )

    def _md_general_information(self, data: Dict[str, Any]) -> str:
        """Render general information as markdown."""
        sec = self._section_general_information(data)
        return (
            f"## {sec['title']}\n*{sec['annex_reference']}*\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Manufacturer | {sec['manufacturer_name']} |\n"
            f"| Battery ID | {sec['battery_unique_id']} |\n"
            f"| Model | {sec['battery_model']} |\n"
            f"| Type | {sec['battery_type']} |\n"
            f"| Chemistry | {sec['battery_chemistry']} |\n"
            f"| Nominal Voltage | {sec['nominal_voltage_v']} V |\n"
            f"| Rated Capacity | {sec['rated_capacity_kwh']} kWh / "
            f"{sec['rated_capacity_ah']} Ah |\n"
            f"| Weight | {sec['weight_kg']} kg |\n"
            f"| Manufacturing Date | {sec['manufacturing_date']} |\n"
            f"| Manufacturing Location | {sec['manufacturing_location']} |\n"
            f"| Warranty | {sec['warranty_period_years']} years |\n"
            f"| Expected Lifetime | {sec['expected_lifetime_years']} years / "
            f"{sec['expected_lifetime_cycles']} cycles |"
        )

    def _md_carbon_footprint(self, data: Dict[str, Any]) -> str:
        """Render carbon footprint as markdown."""
        sec = self._section_carbon_footprint(data)
        shares = sec["carbon_footprint_share_by_stage"]
        return (
            f"## {sec['title']}\n*{sec['annex_reference']}*\n\n"
            f"**Total CF:** {sec['total_carbon_footprint_kgco2e_per_kwh']:.2f} "
            f"kgCO2e/kWh  \n"
            f"**Performance Class:** {sec['performance_class']}\n\n"
            f"| Lifecycle Stage | Share (%) |\n|----------------|----------:|\n"
            f"| Raw Material Acquisition | {shares['raw_material_acquisition_pct']:.1f}% |\n"
            f"| Main Production | {shares['main_production_pct']:.1f}% |\n"
            f"| Distribution | {shares['distribution_pct']:.1f}% |\n"
            f"| End-of-Life | {shares['end_of_life_pct']:.1f}% |\n\n"
            f"**Methodology:** {sec['methodology_reference']}"
        )

    def _md_supply_chain_dd(self, data: Dict[str, Any]) -> str:
        """Render supply chain due diligence as markdown."""
        sec = self._section_supply_chain_due_diligence(data)
        lines = [
            f"## {sec['title']}\n*{sec['annex_reference']}*\n",
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| DD Policy Available | {'Yes' if sec['dd_policy_available'] else 'No'} |\n"
            f"| OECD Guidance Compliant | "
            f"{'Yes' if sec['oecd_guidance_compliant'] else 'No'} |\n"
            f"| Total Suppliers | {sec['total_suppliers']} |\n"
            f"| Assessed Suppliers | {sec['assessed_suppliers']} "
            f"({sec['assessment_coverage_pct']:.1f}%) |\n"
            f"| Third-Party Audit | "
            f"{'Completed' if sec['third_party_audit_completed'] else 'Pending'} |\n"
            f"| Grievance Mechanism | "
            f"{'Established' if sec['grievance_mechanism_established'] else 'Not yet'} |",
        ]
        if sec["risk_areas_identified"]:
            lines.append("\n**Risk Areas:**")
            for area in sec["risk_areas_identified"]:
                lines.append(f"  - {area}")
        return "\n".join(lines)

    def _md_material_composition(self, data: Dict[str, Any]) -> str:
        """Render material composition as markdown."""
        sec = self._section_material_composition(data)
        lines = [
            f"## {sec['title']}\n*{sec['annex_reference']}*\n",
            "| Material | Mass (kg) | Mass % | Hazardous | CRM | Recycled % |",
            "|----------|----------:|-------:|:---------:|:---:|----------:|",
        ]
        for mat in sec["materials"]:
            haz = "Yes" if mat["is_hazardous"] else "No"
            crm = "Yes" if mat["is_critical_raw_material"] else "No"
            lines.append(
                f"| {mat['name']} | {mat['mass_kg']:.3f} | "
                f"{mat['mass_percentage']:.1f}% | {haz} | {crm} | "
                f"{mat['recycled_content_pct']:.1f}% |"
            )
        if sec["hazardous_substances"]:
            lines.append(f"\n**Hazardous Substances:** {sec['hazardous_count']}")
            for h in sec["hazardous_substances"]:
                lines.append(
                    f"  - {h['name']} (CAS: {h['cas_number']}) - "
                    f"{h['concentration_pct']:.3f}%"
                )
        if sec["critical_raw_materials"]:
            lines.append(
                f"\n**Critical Raw Materials:** {sec['critical_raw_material_count']}"
            )
            for crm in sec["critical_raw_materials"]:
                lines.append(
                    f"  - {crm['name']}: {crm['mass_kg']:.3f} kg "
                    f"(source: {crm['source_country']})"
                )
        return "\n".join(lines)

    def _md_performance_durability(self, data: Dict[str, Any]) -> str:
        """Render performance and durability as markdown."""
        sec = self._section_performance_durability(data)
        return (
            f"## {sec['title']}\n*{sec['annex_reference']}*\n\n"
            f"| Parameter | Value |\n|-----------|------:|\n"
            f"| Rated Capacity | {sec['rated_capacity_kwh']} kWh |\n"
            f"| Nominal Voltage | {sec['nominal_voltage_v']} V |\n"
            f"| Voltage Range | {sec['minimum_voltage_v']}-{sec['maximum_voltage_v']} V |\n"
            f"| Original Power | {sec['original_power_w']} W |\n"
            f"| Round-Trip Efficiency | {sec['round_trip_efficiency_pct']:.1f}% |\n"
            f"| Expected Cycle Life | {sec['expected_lifetime_cycles']} cycles |\n"
            f"| Capacity Fade | {sec['capacity_fade_pct']:.1f}% |\n"
            f"| Power Fade | {sec['power_fade_pct']:.1f}% |\n"
            f"| Internal Resistance | {sec['internal_resistance_ohm']:.4f} Ohm |\n"
            f"| Temperature Range | {sec['temperature_range_min_c']} to "
            f"{sec['temperature_range_max_c']} C |\n"
            f"| State of Health | {sec['state_of_health_pct']:.1f}% |\n"
            f"| State of Charge | {sec['state_of_charge_pct']:.1f}% |"
        )

    def _md_end_of_life(self, data: Dict[str, Any]) -> str:
        """Render end-of-life information as markdown."""
        sec = self._section_end_of_life(data)
        lines = [
            f"## {sec['title']}\n*{sec['annex_reference']}*\n",
            f"- **Collection Scheme:** {sec['collection_scheme_info']}",
            f"- **Return Points:** {sec['return_point_info']}",
            f"- **Dismantling Instructions:** {sec['dismantling_instructions_url']}",
            f"- **Second Life Applicable:** "
            f"{'Yes' if sec['second_life_applicable'] else 'No'}",
            f"- **Recyclability:** {sec['recyclability_assessment']}",
            f"- **Recycling Efficiency Target:** "
            f"{sec['recycling_efficiency_target_pct']:.1f}%",
        ]
        if sec["waste_prevention_measures"]:
            lines.append("\n**Waste Prevention Measures:**")
            for measure in sec["waste_prevention_measures"]:
                lines.append(f"  - {measure}")
        if sec["material_recovery_targets"]:
            lines.append("\n**Material Recovery Targets:**")
            for mat_name, target in sec["material_recovery_targets"].items():
                lines.append(f"  - {mat_name}: {target}%")
        return "\n".join(lines)

    def _md_qr_code_data(self, data: Dict[str, Any]) -> str:
        """Render QR code data as markdown."""
        sec = self._section_qr_code_data(data)
        return (
            f"## {sec['title']}\n*{sec['article_reference']}*\n\n"
            f"**Battery ID:** {sec['battery_unique_id']}  \n"
            f"**Passport URL:** {sec['passport_url']}  \n"
            f"**Payload Hash:** `{sec['qr_payload_hash'][:16]}...`\n\n"
            f"```json\n{sec['qr_payload_json']}\n```\n\n"
            f"*{sec['access_rights_note']}*"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n*Battery Passport generated by PACK-020 Battery Passport Prep Pack "
            f"on {ts}*\n"
            f"*Regulation: EU Battery Regulation (EU) 2023/1542, Annex XIII*"
        )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1100px;margin:auto}"
            "h1{color:#0d47a1;border-bottom:2px solid #0d47a1;padding-bottom:.3em}"
            "h2{color:#1565c0;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e3f2fd}"
            ".pass{color:#2e7d32;font-weight:bold}"
            ".fail{color:#c62828;font-weight:bold}"
            ".qr-block{background:#f5f5f5;padding:1em;border-radius:4px;"
            "font-family:monospace;font-size:0.9em}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Battery Passport</h1>\n"
            f"<p>EU Battery Regulation (EU) 2023/1542 - Annex XIII</p>\n"
            f"<p><strong>ID:</strong> {data.get('battery_unique_id', '')} | "
            f"<strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('battery_model', '')}</p>"
        )

    def _html_general_information(self, data: Dict[str, Any]) -> str:
        """Render general information HTML."""
        sec = self._section_general_information(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table>"
            f"<tr><th>Manufacturer</th><td>{sec['manufacturer_name']}</td></tr>"
            f"<tr><th>Battery ID</th><td>{sec['battery_unique_id']}</td></tr>"
            f"<tr><th>Model</th><td>{sec['battery_model']}</td></tr>"
            f"<tr><th>Chemistry</th><td>{sec['battery_chemistry']}</td></tr>"
            f"<tr><th>Capacity</th><td>{sec['rated_capacity_kwh']} kWh</td></tr>"
            f"<tr><th>Weight</th><td>{sec['weight_kg']} kg</td></tr>"
            f"</table>"
        )

    def _html_carbon_footprint(self, data: Dict[str, Any]) -> str:
        """Render carbon footprint HTML."""
        sec = self._section_carbon_footprint(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: <strong>"
            f"{sec['total_carbon_footprint_kgco2e_per_kwh']:.2f} kgCO2e/kWh</strong> | "
            f"Class: <strong>{sec['performance_class']}</strong></p>"
        )

    def _html_supply_chain_dd(self, data: Dict[str, Any]) -> str:
        """Render supply chain due diligence HTML."""
        sec = self._section_supply_chain_due_diligence(data)
        oecd_cls = "pass" if sec["oecd_guidance_compliant"] else "fail"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{oecd_cls}'>OECD Compliance: "
            f"{'Yes' if sec['oecd_guidance_compliant'] else 'No'}</p>\n"
            f"<p>Suppliers assessed: {sec['assessed_suppliers']}/{sec['total_suppliers']} "
            f"({sec['assessment_coverage_pct']:.1f}%)</p>"
        )

    def _html_material_composition(self, data: Dict[str, Any]) -> str:
        """Render material composition HTML."""
        sec = self._section_material_composition(data)
        rows = "".join(
            f"<tr><td>{m['name']}</td><td>{m['mass_kg']:.3f}</td>"
            f"<td>{m['mass_percentage']:.1f}%</td>"
            f"<td>{m['recycled_content_pct']:.1f}%</td></tr>"
            for m in sec["materials"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Material</th><th>Mass (kg)</th><th>%</th>"
            f"<th>Recycled %</th></tr>{rows}</table>"
        )

    def _html_performance_durability(self, data: Dict[str, Any]) -> str:
        """Render performance and durability HTML."""
        sec = self._section_performance_durability(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table>"
            f"<tr><th>Capacity</th><td>{sec['rated_capacity_kwh']} kWh</td></tr>"
            f"<tr><th>Voltage</th><td>{sec['nominal_voltage_v']} V</td></tr>"
            f"<tr><th>Cycle Life</th><td>{sec['expected_lifetime_cycles']}</td></tr>"
            f"<tr><th>Efficiency</th><td>{sec['round_trip_efficiency_pct']:.1f}%</td></tr>"
            f"<tr><th>SoH</th><td>{sec['state_of_health_pct']:.1f}%</td></tr>"
            f"</table>"
        )

    def _html_end_of_life(self, data: Dict[str, Any]) -> str:
        """Render end-of-life HTML."""
        sec = self._section_end_of_life(data)
        second = "Yes" if sec["second_life_applicable"] else "No"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table>"
            f"<tr><th>Second Life</th><td>{second}</td></tr>"
            f"<tr><th>Recycling Target</th>"
            f"<td>{sec['recycling_efficiency_target_pct']:.1f}%</td></tr>"
            f"</table>"
        )

    def _html_qr_code_data(self, data: Dict[str, Any]) -> str:
        """Render QR code data HTML."""
        sec = self._section_qr_code_data(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p><strong>Passport URL:</strong> "
            f"<a href='{sec['passport_url']}'>{sec['passport_url']}</a></p>\n"
            f"<div class='qr-block'>{sec['qr_payload_json']}</div>"
        )
