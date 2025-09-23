from typing import Dict, Any, List, Optional
from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow
from greenlang.agents import (
    FuelAgent,
    CarbonAgent,
    InputValidatorAgent,
    ReportAgent,
    BenchmarkAgent,
    BaseAgent,
    # Enhanced Agents
    BoilerAgent,
    GridFactorAgent,
    BuildingProfileAgent,
    IntensityAgent,
    RecommendationAgent,
    # Climatenza AI Agents
    SiteInputAgent,
    SolarResourceAgent,
    LoadProfileAgent,
    FieldLayoutAgent,
    EnergyBalanceAgent,
)


class GreenLangClient:
    """Python SDK client for GreenLang"""

    def __init__(self):
        self.orchestrator = Orchestrator()
        self._register_default_agents()

    def _register_default_agents(self):
        # Core agents
        self.orchestrator.register_agent("validator", InputValidatorAgent())
        self.orchestrator.register_agent("fuel", FuelAgent())
        self.orchestrator.register_agent("carbon", CarbonAgent())
        self.orchestrator.register_agent("report", ReportAgent())
        self.orchestrator.register_agent("benchmark", BenchmarkAgent())

        # Enhanced agents
        self.orchestrator.register_agent("boiler", BoilerAgent())
        self.orchestrator.register_agent("grid_factor", GridFactorAgent())
        self.orchestrator.register_agent("building_profile", BuildingProfileAgent())
        self.orchestrator.register_agent("intensity", IntensityAgent())
        self.orchestrator.register_agent("recommendation", RecommendationAgent())

        # Climatenza AI agents
        self.orchestrator.register_agent("SiteInputAgent", SiteInputAgent())
        self.orchestrator.register_agent("SolarResourceAgent", SolarResourceAgent())
        self.orchestrator.register_agent("LoadProfileAgent", LoadProfileAgent())
        self.orchestrator.register_agent("FieldLayoutAgent", FieldLayoutAgent())
        self.orchestrator.register_agent("EnergyBalanceAgent", EnergyBalanceAgent())

    def register_agent(self, agent_id: str, agent: BaseAgent):
        self.orchestrator.register_agent(agent_id, agent)

    def register_workflow(self, workflow_id: str, workflow: Workflow):
        self.orchestrator.register_workflow(workflow_id, workflow)

    def execute_workflow(
        self, workflow_id: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.orchestrator.execute_workflow(workflow_id, input_data)

    def execute_agent(
        self, agent_id: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = self.orchestrator.execute_single_agent(agent_id, input_data)
        # execute_single_agent already returns a dict, no need to call model_dump()
        return result

    def calculate_carbon_footprint(
        self, fuels: List[Dict], building_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        workflow = self._create_carbon_workflow()
        self.orchestrator.register_workflow("carbon_calc", workflow)

        input_data = {"fuels": fuels, "building_info": building_info or {}}

        return self.orchestrator.execute_workflow("carbon_calc", input_data)

    def _create_carbon_workflow(self) -> Workflow:
        from greenlang.core.workflow import WorkflowBuilder

        builder = WorkflowBuilder("carbon_footprint", "Calculate carbon footprint")

        builder.add_step("validate", "validator")
        builder.add_step("calculate_fuels", "fuel")
        builder.add_step("aggregate", "carbon")
        builder.add_step("report", "report")

        return builder.build()

    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.execute_agent("validator", data)

    def calculate_emissions(
        self, fuel_type: str, consumption: float, unit: str, region: str = "US"
    ) -> Dict[str, Any]:
        input_data = {
            "fuel_type": fuel_type,
            "consumption": consumption,
            "unit": unit,
            "region": region,
        }
        return self.execute_agent("fuel", input_data)

    def aggregate_emissions(self, emissions_list: List[Dict]) -> Dict[str, Any]:
        input_data = {"emissions": emissions_list}
        return self.execute_agent("carbon", input_data)

    def generate_report(
        self,
        carbon_data: Dict,
        format: str = "text",
        building_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        input_data = {
            "carbon_data": carbon_data,
            "format": format,
            "building_info": building_info or {},
        }
        return self.execute_agent("report", input_data)

    def benchmark_emissions(
        self,
        total_emissions_kg: float,
        building_area: float,
        building_type: str = "commercial_office",
        period_months: int = 12,
    ) -> Dict[str, Any]:
        input_data = {
            "total_emissions_kg": total_emissions_kg,
            "building_area": building_area,
            "building_type": building_type,
            "period_months": period_months,
        }
        return self.execute_agent("benchmark", input_data)

    # Climatenza AI Methods
    def run_solar_feasibility(self, site_config_path: str) -> Dict[str, Any]:
        """Run a complete solar thermal feasibility analysis"""
        workflow_path = "climatenza_app/gl_workflows/feasibility_base.yaml"
        workflow = Workflow.from_yaml(workflow_path)
        self.orchestrator.register_workflow("climatenza", workflow)

        input_data = {"inputs": {"site_file": site_config_path}}

        return self.orchestrator.execute_workflow("climatenza", input_data)

    def calculate_solar_field_size(
        self, annual_demand_gwh: float, solar_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate required solar collector field size"""
        input_data = {
            "total_annual_demand_gwh": annual_demand_gwh,
            "solar_config": solar_config,
        }
        return self.execute_agent("FieldLayoutAgent", input_data)

    def simulate_energy_balance(
        self, solar_data: str, load_data: str, aperture_area: float
    ) -> Dict[str, Any]:
        """Run hourly energy balance simulation"""
        input_data = {
            "solar_resource_df_json": solar_data,
            "load_profile_df_json": load_data,
            "required_aperture_area_m2": aperture_area,
        }
        return self.execute_agent("EnergyBalanceAgent", input_data)

    def get_solar_resource(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetch solar resource data for a location"""
        input_data = {"lat": lat, "lon": lon}
        return self.execute_agent("SolarResourceAgent", input_data)

    def list_agents(self) -> List[str]:
        return self.orchestrator.list_agents()

    def list_workflows(self) -> List[str]:
        return self.orchestrator.list_workflows()

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self.orchestrator.get_agent_info(agent_id)

    def get_workflow_info(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        return self.orchestrator.get_workflow_info(workflow_id)

    def get_execution_history(self) -> List[Dict[str, Any]]:
        return self.orchestrator.get_execution_history()

    def clear_history(self):
        self.orchestrator.clear_history()
