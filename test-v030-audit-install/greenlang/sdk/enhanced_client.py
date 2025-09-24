from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import os
from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow, WorkflowBuilder
from greenlang.agents import (
    FuelAgent,
    CarbonAgent,
    InputValidatorAgent,
    ReportAgent,
    BenchmarkAgent,
    GridFactorAgent,
    BuildingProfileAgent,
    IntensityAgent,
    RecommendationAgent,
    BoilerAgent,
)
from greenlang.data.models import (
    BuildingInput,
)


class GreenLangClient:
    """Enhanced Python SDK client for GreenLang with global support"""

    def __init__(self, region: str = "US", verbose: bool = False):
        """
        Initialize GreenLang client

        Args:
            region: Country/region code (US, IN, EU, CN, JP, BR, KR, etc.)
            verbose: Enable verbose logging
        """
        self.region = region.upper()
        self.verbose = verbose
        self.orchestrator = Orchestrator()
        self._register_all_agents()
        self._load_config()

    def _register_all_agents(self):
        """Register all available agents"""
        self.orchestrator.register_agent("validator", InputValidatorAgent())
        self.orchestrator.register_agent("fuel", FuelAgent())
        self.orchestrator.register_agent("carbon", CarbonAgent())
        self.orchestrator.register_agent("report", ReportAgent())
        self.orchestrator.register_agent("benchmark", BenchmarkAgent())
        self.orchestrator.register_agent("grid_factor", GridFactorAgent())
        self.orchestrator.register_agent("building_profile", BuildingProfileAgent())
        self.orchestrator.register_agent("intensity", IntensityAgent())
        self.orchestrator.register_agent("recommendation", RecommendationAgent())
        self.orchestrator.register_agent("boiler", BoilerAgent())

    def _load_config(self):
        """Load configuration from environment or config file"""
        # Check for environment variables
        self.region = os.getenv("GREENLANG_REGION", self.region)
        self.report_format = os.getenv("GREENLANG_REPORT_FORMAT", "markdown")

        # Load from .greenlang.json if exists
        config_path = Path.home() / ".greenlang.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                self.region = config.get("region", self.region)
                self.report_format = config.get("report_format", self.report_format)

    # Core Methods
    def calculate_emissions(
        self,
        fuel_type: str,
        consumption: float,
        unit: str,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a single fuel type

        Args:
            fuel_type: Type of fuel (electricity, natural_gas, diesel, etc.)
            consumption: Amount consumed
            unit: Unit of consumption (kWh, therms, liters, etc.)
            region: Override default region

        Returns:
            Dict containing emissions data
        """
        # Get emission factor
        factor_result = self.orchestrator.execute_single_agent(
            "grid_factor",
            {"country": region or self.region, "fuel_type": fuel_type, "unit": unit},
        )

        if not factor_result["success"]:
            return {
                "success": False,
                "error": factor_result.get("error", "Failed to get emission factor"),
            }

        emission_factor = factor_result["data"]["emission_factor"]
        co2e_emissions_kg = consumption * emission_factor

        return {
            "success": True,
            "data": {
                "fuel_type": fuel_type,
                "consumption": consumption,
                "unit": unit,
                "emission_factor": emission_factor,
                "co2e_emissions_kg": co2e_emissions_kg,
                "co2e_emissions_tons": co2e_emissions_kg / 1000,
                "region": region or self.region,
            },
        }

    def calculate_boiler_emissions(
        self,
        fuel_type: str,
        thermal_output: float,
        output_unit: str = "kWh",
        efficiency: float = 0.85,
        boiler_type: str = "standard",
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate emissions from boiler operations

        Args:
            fuel_type: Type of fuel used (natural_gas, diesel, biomass, etc.)
            thermal_output: Thermal energy output
            output_unit: Unit of thermal output (kWh, MJ, BTU, etc.)
            efficiency: Boiler efficiency (0.0 to 1.0)
            boiler_type: Type of boiler (condensing, standard, old)
            region: Override default region

        Returns:
            Dict containing boiler emissions data
        """
        boiler_result = self.orchestrator.execute_single_agent(
            "boiler",
            {
                "fuel_type": fuel_type,
                "thermal_output": {"value": thermal_output, "unit": output_unit},
                "efficiency": efficiency,
                "boiler_type": boiler_type,
                "country": region or self.region,
            },
        )

        if not boiler_result["success"]:
            return {
                "success": False,
                "error": boiler_result.get(
                    "error", "Failed to calculate boiler emissions"
                ),
            }

        return {"success": True, "data": boiler_result["data"]}

    def analyze_building(
        self, building_data: Union[Dict, BuildingInput]
    ) -> Dict[str, Any]:
        """
        Comprehensive building analysis with all metrics

        Args:
            building_data: Building data dictionary or BuildingInput model

        Returns:
            Complete analysis including emissions, intensity, benchmarks, and recommendations
        """
        # Convert to dict if needed
        if isinstance(building_data, BuildingInput):
            building_data = building_data.model_dump()

        results = {"success": True, "data": {}}

        # 1. Building Profile Analysis
        profile_result = self.orchestrator.execute_single_agent(
            "building_profile",
            {
                "building_type": building_data["metadata"]["building_type"],
                "area": building_data["metadata"]["area"],
                "area_unit": building_data["metadata"].get("area_unit", "sqft"),
                "occupancy": building_data["metadata"].get("occupancy"),
                "floor_count": building_data["metadata"].get("floor_count"),
                "building_age": building_data["metadata"].get("building_age"),
                "climate_zone": building_data["metadata"].get("climate_zone"),
                "country": building_data["metadata"]["location"]["country"],
            },
        )

        if profile_result["success"]:
            results["data"]["profile"] = profile_result["data"]

        # 2. Calculate Emissions
        emissions_list = []
        emissions_by_source = {}
        total_energy_kwh = 0

        for fuel_type, fuel_data in building_data.get("energy_consumption", {}).items():
            if fuel_type == "solar_pv_generation":
                continue

            emission_result = self.calculate_emissions(
                fuel_type,
                fuel_data["value"],
                fuel_data["unit"],
                building_data["metadata"]["location"]["country"],
            )

            if emission_result["success"]:
                emissions_list.append(emission_result["data"])
                emissions_by_source[fuel_type] = emission_result["data"][
                    "co2e_emissions_kg"
                ]

                # Track total energy
                if fuel_type in ["electricity", "district_heating"]:
                    total_energy_kwh += fuel_data["value"]

        # Account for solar offset
        if "solar_pv_generation" in building_data.get("energy_consumption", {}):
            solar_kwh = building_data["energy_consumption"]["solar_pv_generation"][
                "value"
            ]
            if "electricity" in emissions_by_source and total_energy_kwh > 0:
                reduction_factor = max(0, 1 - (solar_kwh / total_energy_kwh))
                emissions_by_source["electricity"] *= reduction_factor

        # 3. Aggregate Emissions
        carbon_result = self.orchestrator.execute_single_agent(
            "carbon", {"emissions": emissions_list}
        )

        if carbon_result["success"]:
            results["data"]["emissions"] = carbon_result["data"]

        # 4. Calculate Intensity
        total_emissions_kg = sum(emissions_by_source.values())
        intensity_result = self.orchestrator.execute_single_agent(
            "intensity",
            {
                "total_emissions_kg": total_emissions_kg,
                "area": building_data["metadata"]["area"],
                "area_unit": building_data["metadata"].get("area_unit", "sqft"),
                "occupancy": building_data["metadata"].get("occupancy"),
                "floor_count": building_data["metadata"].get("floor_count"),
                "period_months": 12,
                "building_type": building_data["metadata"]["building_type"],
                "country": building_data["metadata"]["location"]["country"],
                "total_energy_kwh": total_energy_kwh,
            },
        )

        if intensity_result["success"]:
            results["data"]["intensity"] = intensity_result["data"]

        # 5. Benchmark
        benchmark_result = self.benchmark_emissions(
            total_emissions_kg,
            building_data["metadata"]["area"],
            building_data["metadata"]["building_type"],
            12,
        )

        if benchmark_result["success"]:
            results["data"]["benchmark"] = benchmark_result["data"]

        # 6. Recommendations
        rec_result = self.orchestrator.execute_single_agent(
            "recommendation",
            {
                "emissions_by_source": emissions_by_source,
                "intensity": results["data"]
                .get("intensity", {})
                .get("intensities", {}),
                "building_type": building_data["metadata"]["building_type"],
                "building_age": building_data["metadata"].get("building_age", 10),
                "performance_rating": results["data"]
                .get("intensity", {})
                .get("performance_rating", "Average"),
                "load_breakdown": results["data"]
                .get("profile", {})
                .get("load_breakdown", {}),
                "country": building_data["metadata"]["location"]["country"],
            },
        )

        if rec_result["success"]:
            results["data"]["recommendations"] = rec_result["data"]

        return results

    def benchmark_emissions(
        self,
        total_emissions_kg: float,
        building_area: float,
        building_type: str = "commercial_office",
        period_months: int = 12,
    ) -> Dict[str, Any]:
        """
        Benchmark emissions against standards

        Args:
            total_emissions_kg: Total emissions in kg CO2e
            building_area: Building area in sqft
            building_type: Type of building
            period_months: Period in months

        Returns:
            Benchmark analysis
        """
        input_data = {
            "total_emissions_kg": total_emissions_kg,
            "building_area": building_area,
            "building_type": building_type,
            "period_months": period_months,
        }
        result = self.orchestrator.execute_single_agent("benchmark", input_data)
        return {
            "success": result["success"],
            "data": result["data"] if result["success"] else {},
            "error": result.get("error"),
        }

    def get_recommendations(
        self,
        building_type: str,
        performance_rating: str,
        country: Optional[str] = None,
        building_age: int = 10,
    ) -> Dict[str, Any]:
        """
        Get optimization recommendations

        Args:
            building_type: Type of building
            performance_rating: Current performance (Excellent/Good/Average/Poor)
            country: Country/region
            building_age: Age of building in years

        Returns:
            Recommendations and roadmap
        """
        result = self.orchestrator.execute_single_agent(
            "recommendation",
            {
                "building_type": building_type,
                "performance_rating": performance_rating,
                "country": country or self.region,
                "building_age": building_age,
            },
        )

        return {
            "success": result["success"],
            "data": result["data"] if result["success"] else {},
            "error": result.get("error"),
        }

    def get_emission_factor(
        self, fuel_type: str, unit: str, country: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get emission factor for a fuel type

        Args:
            fuel_type: Type of fuel
            unit: Unit of measurement
            country: Country/region

        Returns:
            Emission factor data
        """
        result = self.orchestrator.execute_single_agent(
            "grid_factor",
            {"country": country or self.region, "fuel_type": fuel_type, "unit": unit},
        )

        return {
            "success": result["success"],
            "data": result["data"] if result["success"] else {},
            "error": result.get("error"),
        }

    def calculate_intensity(
        self,
        total_emissions_kg: float,
        area: float,
        occupancy: Optional[int] = None,
        floor_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate various intensity metrics

        Args:
            total_emissions_kg: Total emissions
            area: Building area in sqft
            occupancy: Number of occupants
            floor_count: Number of floors

        Returns:
            Intensity metrics
        """
        result = self.orchestrator.execute_single_agent(
            "intensity",
            {
                "total_emissions_kg": total_emissions_kg,
                "area": area,
                "area_unit": "sqft",
                "occupancy": occupancy,
                "floor_count": floor_count,
                "period_months": 12,
            },
        )

        return {
            "success": result["success"],
            "data": result["data"] if result["success"] else {},
            "error": result.get("error"),
        }

    def aggregate_emissions(self, emissions_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate multiple emission sources"""
        result = self.orchestrator.execute_single_agent(
            "carbon", {"emissions": emissions_list}
        )
        return {
            "success": result["success"],
            "data": result["data"] if result["success"] else {},
            "error": result.get("error"),
        }

    def generate_report(
        self, analysis_results: Dict, format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report

        Args:
            analysis_results: Results from analyze_building
            format: Output format (text/markdown/json)

        Returns:
            Formatted report
        """
        result = self.orchestrator.execute_single_agent(
            "report",
            {
                "carbon_data": analysis_results.get("data", {}).get("emissions", {}),
                "format": format,
                "building_info": analysis_results.get("data", {}).get("profile", {}),
                "intensity_data": analysis_results.get("data", {}).get("intensity", {}),
                "benchmark_data": analysis_results.get("data", {}).get("benchmark", {}),
                "recommendations": analysis_results.get("data", {}).get(
                    "recommendations", {}
                ),
            },
        )

        return {
            "success": result["success"],
            "data": result["data"] if result["success"] else {},
            "error": result.get("error"),
        }

    # Workflow Methods
    def create_workflow(self, name: str, description: str) -> WorkflowBuilder:
        """Create a new workflow builder"""
        return WorkflowBuilder(name, description)

    def register_workflow(self, workflow_id: str, workflow: Union[Workflow, str, Path]):
        """
        Register a workflow

        Args:
            workflow_id: Unique workflow identifier
            workflow: Workflow object, YAML string, or path to YAML file
        """
        if isinstance(workflow, (str, Path)):
            if Path(workflow).exists():
                workflow = Workflow.from_yaml(str(workflow))
            else:
                # Assume it's YAML content
                import yaml

                workflow_dict = yaml.safe_load(workflow)
                workflow = Workflow.from_dict(workflow_dict)

        self.orchestrator.register_workflow(workflow_id, workflow)

    def execute_workflow(
        self, workflow_id: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a registered workflow"""
        return self.orchestrator.execute_workflow(workflow_id, input_data)

    # Utility Methods
    def list_agents(self) -> List[str]:
        """List all registered agents"""
        return self.orchestrator.list_agents()

    def list_workflows(self) -> List[str]:
        """List all registered workflows"""
        return self.orchestrator.list_workflows()

    def execute_agent(
        self, agent_id: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single agent with input data"""
        result = self.orchestrator.execute_single_agent(agent_id, input_data)
        return result

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered agent"""
        if agent_id not in self.orchestrator.agents:
            return None

        agent = self.orchestrator.agents[agent_id]

        # Try to get config first, if not available use defaults
        if hasattr(agent, "config"):
            return {
                "id": agent_id,
                "name": agent.config.name,
                "description": agent.config.description,
                "version": agent.config.version,
                "enabled": agent.config.enabled,
            }
        else:
            # Fallback for agents without config
            descriptions = {
                "validator": "Validates input data for emissions calculations",
                "fuel": "Calculates emissions based on fuel consumption",
                "boiler": "Calculates emissions from boilers and thermal systems",
                "carbon": "Aggregates emissions and provides carbon footprint",
                "report": "Generates carbon footprint reports",
                "benchmark": "Compares emissions against industry benchmarks",
                "grid_factor": "Retrieves country-specific emission factors",
                "building_profile": "Categorizes buildings and expected performance",
                "intensity": "Calculates emission intensity metrics",
                "recommendation": "Provides optimization recommendations",
            }

            return {
                "id": agent_id,
                "name": agent.__class__.__name__,
                "description": descriptions.get(
                    agent_id, "Agent for climate calculations"
                ),
                "version": getattr(agent, "version", "0.0.1"),
                "enabled": True,
            }

    def get_supported_countries(self) -> List[str]:
        """Get list of supported countries"""
        grid_agent = GridFactorAgent()
        return grid_agent.get_available_countries()

    def get_supported_fuel_types(self, country: Optional[str] = None) -> List[str]:
        """Get supported fuel types for a country"""
        grid_agent = GridFactorAgent()
        return grid_agent.get_available_fuel_types(country or self.region)

    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data"""
        result = self.orchestrator.execute_single_agent("validator", data)
        return result

    def validate_building_data(self, building_data: Dict) -> Dict[str, Any]:
        """Validate building data structure"""
        result = self.orchestrator.execute_single_agent("validator", building_data)
        return {
            "success": result["success"],
            "data": result["data"] if result["success"] else {},
            "error": result.get("error"),
        }

    # Batch Processing
    def analyze_portfolio(self, buildings: List[Dict]) -> Dict[str, Any]:
        """
        Analyze multiple buildings (portfolio analysis)

        Args:
            buildings: List of building data dictionaries

        Returns:
            Portfolio-level analysis
        """
        portfolio_results = []
        total_emissions = 0
        total_area = 0

        for i, building in enumerate(buildings):
            result = self.analyze_building(building)
            if result["success"]:
                portfolio_results.append(
                    {
                        "building_id": building.get("id", f"building_{i+1}"),
                        "analysis": result["data"],
                    }
                )

                # Aggregate portfolio metrics
                if "emissions" in result["data"]:
                    total_emissions += result["data"]["emissions"].get(
                        "total_co2e_kg", 0
                    )
                total_area += building["metadata"]["area"]

        # Portfolio-level metrics
        portfolio_intensity = total_emissions / total_area if total_area > 0 else 0

        return {
            "success": True,
            "data": {
                "buildings": portfolio_results,
                "portfolio_metrics": {
                    "total_buildings": len(buildings),
                    "total_emissions_kg": total_emissions,
                    "total_emissions_tons": total_emissions / 1000,
                    "total_area_sqft": total_area,
                    "average_intensity": portfolio_intensity,
                    "region": self.region,
                },
            },
        }

    # Export/Import Methods
    def export_analysis(
        self, analysis_results: Dict, filepath: str, format: str = "json"
    ):
        """
        Export analysis results to file

        Args:
            analysis_results: Results from analyze_building
            filepath: Output file path
            format: Export format (json/csv/excel)
        """
        import json

        if format == "json":
            with open(filepath, "w") as f:
                json.dump(analysis_results, f, indent=2, default=str)
        elif format == "csv":
            # Flatten and export as CSV
            try:
                import pandas as pd
            except ImportError:
                raise ImportError(
                    "pandas is required for CSV export. "
                    "Install it with: pip install greenlang[analytics]"
                )

            # Flatten nested dictionary
            flattened = self._flatten_dict(analysis_results)
            df = pd.DataFrame([flattened])
            df.to_csv(filepath, index=False)
        elif format == "excel":
            try:
                import pandas as pd
            except ImportError:
                raise ImportError(
                    "pandas is required for Excel export. "
                    "Install it with: pip install greenlang[analytics]"
                )

            # Create multiple sheets for different sections
            with pd.ExcelWriter(filepath) as writer:
                if "emissions" in analysis_results.get("data", {}):
                    emissions_df = pd.DataFrame([analysis_results["data"]["emissions"]])
                    emissions_df.to_excel(writer, sheet_name="Emissions", index=False)

                if "intensity" in analysis_results.get("data", {}):
                    intensity_df = pd.DataFrame(
                        [analysis_results["data"]["intensity"]["intensities"]]
                    )
                    intensity_df.to_excel(writer, sheet_name="Intensity", index=False)

                if "recommendations" in analysis_results.get("data", {}):
                    recs = analysis_results["data"]["recommendations"].get(
                        "recommendations", []
                    )
                    if recs:
                        recs_df = pd.DataFrame(recs)
                        recs_df.to_excel(
                            writer, sheet_name="Recommendations", index=False
                        )

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
