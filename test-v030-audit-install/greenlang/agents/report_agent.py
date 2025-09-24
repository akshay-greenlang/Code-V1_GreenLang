from typing import Any, Dict
from datetime import datetime
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig


class ReportAgent(BaseAgent):
    """Agent for generating carbon footprint reports in various formats"""

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="ReportAgent", description="Generates carbon footprint reports"
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        report_format = input_data.get("format", "text")
        carbon_data = input_data.get("carbon_data", {})
        building_info = input_data.get("building_info", {})
        period_info = input_data.get("period", {})

        if report_format == "markdown":
            report = self._generate_markdown_report(
                carbon_data, building_info, period_info
            )
        elif report_format == "json":
            report = self._generate_json_report(carbon_data, building_info, period_info)
        else:
            report = self._generate_text_report(carbon_data, building_info, period_info)

        return AgentResult(
            success=True,
            data={
                "report": report,
                "format": report_format,
                "generated_at": datetime.now().isoformat(),
            },
            metadata={"agent": "ReportAgent", "report_format": report_format},
        )

    def _generate_text_report(
        self, carbon_data: Dict, building_info: Dict, period_info: Dict
    ) -> str:
        report = []
        report.append("=" * 60)
        report.append("CARBON FOOTPRINT REPORT")
        report.append("=" * 60)
        report.append("")

        if building_info:
            report.append("BUILDING INFORMATION")
            report.append("-" * 40)
            if "type" in building_info:
                report.append(f"Building Type: {building_info['type']}")
            if "area" in building_info:
                report.append(
                    f"Building Area: {building_info['area']:,.0f} {building_info.get('area_unit', 'sqft')}"
                )
            if "occupancy" in building_info:
                report.append(f"Occupancy: {building_info['occupancy']} people")
            report.append("")

        if period_info:
            report.append("REPORTING PERIOD")
            report.append("-" * 40)
            if "start_date" in period_info and "end_date" in period_info:
                report.append(
                    f"Period: {period_info['start_date']} to {period_info['end_date']}"
                )
            elif "duration" in period_info:
                report.append(
                    f"Duration: {period_info['duration']} {period_info.get('duration_unit', 'month')}(s)"
                )
            report.append("")

        report.append("EMISSIONS SUMMARY")
        report.append("-" * 40)

        total_co2e = carbon_data.get("total_co2e_tons", 0)
        report.append(f"Total Emissions: {total_co2e:.3f} metric tons CO2e")
        report.append(
            f"                 {carbon_data.get('total_co2e_kg', 0):.2f} kg CO2e"
        )
        report.append("")

        if "carbon_intensity" in carbon_data:
            intensity = carbon_data["carbon_intensity"]
            if "per_sqft" in intensity:
                report.append(
                    f"Carbon Intensity: {intensity['per_sqft']:.2f} kg CO2e/sqft"
                )
            if "per_person" in intensity:
                report.append(
                    f"                  {intensity['per_person']:.2f} kg CO2e/person"
                )
            report.append("")

        if "emissions_breakdown" in carbon_data:
            report.append("EMISSIONS BY SOURCE")
            report.append("-" * 40)
            for item in carbon_data["emissions_breakdown"]:
                source = item["source"]
                co2e_tons = item["co2e_tons"]
                percentage = item["percentage"]
                report.append(f"{source:20} {co2e_tons:8.3f} tons ({percentage:5.1f}%)")
            report.append("")

        report.append("=" * 60)
        report.append(
            f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        return "\n".join(report)

    def _generate_markdown_report(
        self, carbon_data: Dict, building_info: Dict, period_info: Dict
    ) -> str:
        report = []
        report.append("# Carbon Footprint Report")
        report.append("")

        if building_info:
            report.append("## Building Information")
            report.append("")
            if "type" in building_info:
                report.append(f"- **Building Type:** {building_info['type']}")
            if "area" in building_info:
                report.append(
                    f"- **Building Area:** {building_info['area']:,.0f} {building_info.get('area_unit', 'sqft')}"
                )
            if "occupancy" in building_info:
                report.append(f"- **Occupancy:** {building_info['occupancy']} people")
            report.append("")

        if period_info:
            report.append("## Reporting Period")
            report.append("")
            if "start_date" in period_info and "end_date" in period_info:
                report.append(
                    f"- **Period:** {period_info['start_date']} to {period_info['end_date']}"
                )
            elif "duration" in period_info:
                report.append(
                    f"- **Duration:** {period_info['duration']} {period_info.get('duration_unit', 'month')}(s)"
                )
            report.append("")

        report.append("## Emissions Summary")
        report.append("")

        total_co2e = carbon_data.get("total_co2e_tons", 0)
        report.append(f"### Total Emissions: **{total_co2e:.3f}** metric tons CO2e")
        report.append("")

        if "carbon_intensity" in carbon_data:
            report.append("### Carbon Intensity")
            report.append("")
            intensity = carbon_data["carbon_intensity"]
            if "per_sqft" in intensity:
                report.append(
                    f"- **Per Square Foot:** {intensity['per_sqft']:.2f} kg CO2e/sqft"
                )
            if "per_person" in intensity:
                report.append(
                    f"- **Per Person:** {intensity['per_person']:.2f} kg CO2e/person"
                )
            report.append("")

        if "emissions_breakdown" in carbon_data:
            report.append("## Emissions Breakdown")
            report.append("")
            report.append("| Source | Emissions (tons) | Percentage |")
            report.append("|--------|-----------------|------------|")
            for item in carbon_data["emissions_breakdown"]:
                source = item["source"]
                co2e_tons = item["co2e_tons"]
                percentage = item["percentage"]
                report.append(f"| {source} | {co2e_tons:.3f} | {percentage:.1f}% |")
            report.append("")

        report.append("---")
        report.append(
            f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        )

        return "\n".join(report)

    def _generate_json_report(
        self, carbon_data: Dict, building_info: Dict, period_info: Dict
    ) -> Dict:
        return {
            "report_type": "carbon_footprint",
            "generated_at": datetime.now().isoformat(),
            "building_info": building_info,
            "period": period_info,
            "emissions": {
                "total": {
                    "value": carbon_data.get("total_co2e_tons", 0),
                    "unit": "metric_tons_co2e",
                },
                "breakdown": carbon_data.get("emissions_breakdown", []),
                "intensity": carbon_data.get("carbon_intensity", {}),
            },
            "metadata": {"version": "0.0.1", "agent": "ReportAgent"},
        }
