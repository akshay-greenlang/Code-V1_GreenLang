import re
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import WorkflowBuilder
from greenlang.agents import (
    FuelAgent,
    CarbonAgent,
    InputValidatorAgent,
    ReportAgent,
    BenchmarkAgent,
    BoilerAgent,
    GridFactorAgent,
    BuildingProfileAgent,
    IntensityAgent,
    RecommendationAgent,
)

# Try importing RAG-enhanced assistant
try:
    from greenlang.cli.assistant_rag import RAGAssistant

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AIAssistant:
    """Natural language interface for GreenLang using LLMs"""

    def __init__(self, api_key: Optional[str] = None, use_rag: bool = True):
        # Try to use RAG-enhanced assistant if available
        if RAG_AVAILABLE and use_rag:
            try:
                self._rag_assistant = RAGAssistant(api_key=api_key)
                self._use_rag = True
            except Exception as e:
                print(f"Warning: Could not initialize RAG assistant: {e}")
                self._use_rag = False
        else:
            self._use_rag = False

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.orchestrator = self._setup_orchestrator()

        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.llm_available = True
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                self.llm_available = False
        else:
            self.llm_available = False
            if not OPENAI_AVAILABLE:
                print(
                    "Warning: OpenAI library not installed. Install with: pip install openai"
                )
            if not self.api_key:
                print(
                    "Warning: No OpenAI API key found. Set OPENAI_API_KEY in .env file"
                )

    def _setup_orchestrator(self) -> Orchestrator:
        orchestrator = Orchestrator()
        orchestrator.register_agent("validator", InputValidatorAgent())
        orchestrator.register_agent("fuel", FuelAgent())
        orchestrator.register_agent("boiler", BoilerAgent())
        orchestrator.register_agent("carbon", CarbonAgent())
        orchestrator.register_agent("report", ReportAgent())
        orchestrator.register_agent("benchmark", BenchmarkAgent())
        orchestrator.register_agent("grid_factor", GridFactorAgent())
        orchestrator.register_agent("building_profile", BuildingProfileAgent())
        orchestrator.register_agent("intensity", IntensityAgent())
        orchestrator.register_agent("recommendation", RecommendationAgent())
        return orchestrator

    def process_query(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        """Process user query - either general question or calculation request"""

        # If RAG assistant is available, delegate to it
        if self._use_rag:
            return self._rag_assistant.process_query(
                query, verbose=verbose, use_rag=True
            )

        # Otherwise use standard processing
        # Check if this is a calculation request
        if self._is_calculation_request(query):
            if self.llm_available:
                return self._process_calculation_with_llm(query, verbose)
            else:
                return self._process_with_rules(query, verbose)
        else:
            # This is a general question about emissions
            if self.llm_available:
                return self._answer_general_question(query)
            else:
                return self._answer_with_knowledge_base(query)

    def _is_calculation_request(self, query: str) -> bool:
        """Check if the query is asking for a calculation"""
        calculation_keywords = [
            "calculate",
            "compute",
            "what is the carbon footprint",
            "emissions for",
            "how much co2",
            "carbon emissions",
            "kwh",
            "therms",
            "gallons",
            "liters",
            "consumption",
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in calculation_keywords)

    def _answer_general_question(self, query: str) -> Dict[str, Any]:
        """Answer general questions about emissions using OpenAI"""
        try:
            system_prompt = """You are GreenLang AI Assistant, an expert on carbon emissions, climate change, and environmental sustainability.
            
You have access to emission factors and can answer questions about:
- Emission factors for different fuels and regions
- Carbon footprint calculations and methodologies
- Climate change and environmental impact
- Best practices for reducing emissions
- Industry benchmarks and standards

Known emission factors (these are examples, provide region-specific when asked):
- Electricity: 
  - US Average: 0.385 kg CO2e/kWh
  - India: 0.82 kg CO2e/kWh  
  - EU Average: 0.295 kg CO2e/kWh
  - China: 0.555 kg CO2e/kWh
- Natural Gas: 5.3 kg CO2e/therm (US)
- Diesel: 10.21 kg CO2e/gallon
- Gasoline: 8.887 kg CO2e/gallon
- Propane: 5.76 kg CO2e/gallon

Provide accurate, helpful information. If asked about specific calculations, suggest using the 'gl calc' command for interactive calculations."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.7,
                max_tokens=500,
            )

            answer = response.choices[0].message.content

            return {"success": True, "response": answer}

        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing with OpenAI: {str(e)}",
                "response": self._answer_with_knowledge_base(query)["response"],
            }

    def _answer_with_knowledge_base(self, query: str) -> Dict[str, Any]:
        """Answer questions using built-in knowledge base when OpenAI is not available"""
        query_lower = query.lower()

        # Knowledge base of emission factors
        knowledge_base = {
            "electricity": {
                "us": "0.385 kg CO2e/kWh (US average)",
                "india": "0.82 kg CO2e/kWh (India grid average)",
                "eu": "0.295 kg CO2e/kWh (EU average)",
                "china": "0.555 kg CO2e/kWh (China grid average)",
                "uk": "0.233 kg CO2e/kWh (UK grid average)",
                "default": "0.385 kg CO2e/kWh (varies by region)",
            },
            "natural_gas": "5.3 kg CO2e/therm or 53.06 kg CO2e/MMBtu",
            "diesel": "10.21 kg CO2e/gallon",
            "gasoline": "8.887 kg CO2e/gallon",
            "propane": "5.76 kg CO2e/gallon",
            "coal": "2.86 kg CO2e/kg (anthracite)",
            "jet_fuel": "9.75 kg CO2e/gallon",
        }

        response_parts = []

        # Check for emission factor questions
        if "emission factor" in query_lower or "factor" in query_lower:
            if "electricity" in query_lower:
                if "india" in query_lower:
                    response_parts.append(
                        f"Electricity emission factor in India: {knowledge_base['electricity']['india']}"
                    )
                elif (
                    "us" in query_lower
                    or "united states" in query_lower
                    or "america" in query_lower
                ):
                    response_parts.append(
                        f"Electricity emission factor in US: {knowledge_base['electricity']['us']}"
                    )
                elif "eu" in query_lower or "europe" in query_lower:
                    response_parts.append(
                        f"Electricity emission factor in EU: {knowledge_base['electricity']['eu']}"
                    )
                elif "china" in query_lower:
                    response_parts.append(
                        f"Electricity emission factor in China: {knowledge_base['electricity']['china']}"
                    )
                elif "uk" in query_lower or "britain" in query_lower:
                    response_parts.append(
                        f"Electricity emission factor in UK: {knowledge_base['electricity']['uk']}"
                    )
                else:
                    response_parts.append(
                        f"Electricity emission factor: {knowledge_base['electricity']['default']}"
                    )
                    response_parts.append("\nRegion-specific factors:")
                    for region, factor in knowledge_base["electricity"].items():
                        if region != "default":
                            response_parts.append(f"  - {region.upper()}: {factor}")

            elif "natural gas" in query_lower or "gas" in query_lower:
                response_parts.append(
                    f"Natural gas emission factor: {knowledge_base['natural_gas']}"
                )

            elif "diesel" in query_lower:
                response_parts.append(
                    f"Diesel emission factor: {knowledge_base['diesel']}"
                )

            elif "gasoline" in query_lower or "petrol" in query_lower:
                response_parts.append(
                    f"Gasoline emission factor: {knowledge_base['gasoline']}"
                )

            elif "propane" in query_lower:
                response_parts.append(
                    f"Propane emission factor: {knowledge_base['propane']}"
                )

            elif "coal" in query_lower:
                response_parts.append(f"Coal emission factor: {knowledge_base['coal']}")

            else:
                response_parts.append("Common emission factors:")
                response_parts.append(
                    f"  - Electricity: {knowledge_base['electricity']['default']}"
                )
                response_parts.append(
                    f"  - Natural Gas: {knowledge_base['natural_gas']}"
                )
                response_parts.append(f"  - Diesel: {knowledge_base['diesel']}")
                response_parts.append(f"  - Gasoline: {knowledge_base['gasoline']}")
                response_parts.append(f"  - Propane: {knowledge_base['propane']}")

        # Check for benchmark questions
        elif (
            "benchmark" in query_lower
            or "average" in query_lower
            or "typical" in query_lower
        ):
            response_parts.append("Typical carbon intensity benchmarks for buildings:")
            response_parts.append("  - Excellent: < 5.0 kg CO2e/sqft/year")
            response_parts.append("  - Good: 5.0 - 8.0 kg CO2e/sqft/year")
            response_parts.append("  - Average: 8.0 - 12.0 kg CO2e/sqft/year")
            response_parts.append("  - Below Average: 12.0 - 15.0 kg CO2e/sqft/year")
            response_parts.append("  - Poor: > 15.0 kg CO2e/sqft/year")

        # Check for reduction tips
        elif (
            "reduce" in query_lower
            or "lower" in query_lower
            or "decrease" in query_lower
        ):
            response_parts.append("Tips to reduce carbon emissions:")
            response_parts.append("  1. Switch to renewable energy sources")
            response_parts.append("  2. Improve building insulation")
            response_parts.append("  3. Use energy-efficient appliances")
            response_parts.append("  4. Optimize HVAC systems")
            response_parts.append("  5. Implement smart building controls")
            response_parts.append("  6. Regular maintenance of equipment")

        # Default response if no specific match
        if not response_parts:
            response_parts.append("I can help you with:")
            response_parts.append("  - Emission factors for various fuels")
            response_parts.append("  - Carbon footprint calculations")
            response_parts.append("  - Industry benchmarks")
            response_parts.append("  - Tips for reducing emissions")
            response_parts.append(
                "\nTry asking about specific emission factors or use 'gl calc' for calculations."
            )

        return {"success": True, "response": "\n".join(response_parts)}

    def _process_calculation_with_llm(
        self, query: str, verbose: bool
    ) -> Dict[str, Any]:
        """Process calculation requests with LLM"""
        try:
            prompt = self._create_llm_prompt(query)

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant for GreenLang. Extract structured data from user queries about carbon emissions calculations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            extracted_data = self._parse_llm_response(
                response.choices[0].message.content
            )

            if not extracted_data or not extracted_data.get("fuels"):
                # If no calculation data found, treat as general question
                return self._answer_general_question(query)

            return self._execute_workflow(extracted_data, verbose)

        except Exception as e:
            return {
                "success": False,
                "error": f"LLM processing failed: {str(e)}",
                "response": self._process_with_rules(query, verbose)["response"],
            }

    def _process_with_rules(self, query: str, verbose: bool) -> Dict[str, Any]:
        try:
            extracted_data = self._extract_data_from_query(query)

            if not extracted_data.get("fuels"):
                return {
                    "success": False,
                    "error": "Could not extract fuel consumption data from query",
                    "response": "Please provide fuel consumption data. Example: 'Calculate emissions for 1000 kWh electricity in India'\n\nFor general questions about emission factors, please enable OpenAI API by setting OPENAI_API_KEY in the .env file.",
                }

            result = self._execute_workflow(extracted_data, verbose)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"Error processing request: {str(e)}",
            }

    def _extract_data_from_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        data = {"fuels": [], "building_info": {}, "country": None}

        # Extract country/region - check longer patterns first
        countries = [
            ("united states", "US"),
            ("united kingdom", "UK"),
            ("south korea", "KR"),
            ("india", "IN"),
            ("usa", "US"),
            ("america", "US"),
            ("china", "CN"),
            ("europe", "EU"),
            ("european", "EU"),
            ("japan", "JP"),
            ("brazil", "BR"),
            ("korea", "KR"),
            ("britain", "UK"),
            ("germany", "DE"),
            ("canada", "CA"),
            ("australia", "AU"),
            # Short codes at the end with word boundaries
            (" us ", "US"),
            (" in ", "IN"),
            (" cn ", "CN"),
            (" eu ", "EU"),
            (" jp ", "JP"),
            (" br ", "BR"),
            (" kr ", "KR"),
            (" uk ", "UK"),
            (" de ", "DE"),
            (" ca ", "CA"),
            (" au ", "AU"),
            (" us$", "US"),
            (" in$", "IN"),  # At end of string
        ]

        # Add spaces for boundary checking
        query_check = " " + query_lower + " "

        for country_name, country_code in countries:
            if country_name in query_check:
                data["country"] = country_code
                break

        electricity_pattern = r"(\d+(?:\.\d+)?)\s*(kwh|mwh|gwh)"
        electricity_match = re.search(electricity_pattern, query_lower)
        if electricity_match:
            value = float(electricity_match.group(1))
            unit = electricity_match.group(2)
            # Convert to proper case
            unit_map = {"kwh": "kWh", "mwh": "MWh", "gwh": "GWh"}
            unit = unit_map.get(unit, unit)
            data["fuels"].append(
                {"fuel_type": "electricity", "consumption": value, "unit": unit}
            )

        gas_patterns = [
            (r"(\d+(?:\.\d+)?)\s*therms?\s*(?:of\s*)?(?:natural\s*)?gas", "therms"),
            (r"(\d+(?:\.\d+)?)\s*ccf\s*(?:of\s*)?(?:natural\s*)?gas", "ccf"),
            (
                r"(\d+(?:\.\d+)?)\s*(?:cubic\s*meters?|m3)\s*(?:of\s*)?(?:natural\s*)?gas",
                "m3",
            ),
        ]

        for pattern, unit in gas_patterns:
            match = re.search(pattern, query_lower)
            if match:
                value = float(match.group(1))
                data["fuels"].append(
                    {"fuel_type": "natural_gas", "consumption": value, "unit": unit}
                )
                break

        diesel_pattern = (
            r"(\d+(?:\.\d+)?)\s*(gallons?|liters?|litres?)\s*(?:of\s*)?diesel"
        )
        diesel_match = re.search(diesel_pattern, query_lower)
        if diesel_match:
            value = float(diesel_match.group(1))
            unit = diesel_match.group(2).rstrip("s")
            data["fuels"].append(
                {
                    "fuel_type": "diesel",
                    "consumption": value,
                    "unit": unit if unit != "litre" else "liter",
                }
            )

        area_pattern = (
            r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:sq\.?\s*ft|square\s*feet?|sqft)"
        )
        area_match = re.search(area_pattern, query_lower)
        if area_match:
            area_str = area_match.group(1).replace(",", "")
            data["building_info"]["area"] = float(area_str)
            data["building_info"]["area_unit"] = "sqft"

        occupancy_pattern = r"(\d+)\s*(?:people|persons?|occupants?)"
        occupancy_match = re.search(occupancy_pattern, query_lower)
        if occupancy_match:
            data["building_info"]["occupancy"] = int(occupancy_match.group(1))

        building_types = ["office", "retail", "warehouse", "residential"]
        for building_type in building_types:
            if building_type in query_lower:
                if building_type == "office":
                    data["building_info"]["type"] = "commercial_office"
                else:
                    data["building_info"]["type"] = building_type
                break

        return data

    def _execute_workflow(self, data: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
        try:
            # Simple direct calculation without complex workflow
            from greenlang.sdk import GreenLangClient

            # Use country if specified
            country = data.get("country", "US")
            client = GreenLangClient(region=country)

            emissions_list = []
            for fuel in data.get("fuels", []):
                calc_result = client.calculate_emissions(
                    fuel["fuel_type"], fuel["consumption"], fuel["unit"]
                )
                if calc_result["success"]:
                    emissions_list.append(calc_result["data"])

            if not emissions_list:
                return {
                    "success": False,
                    "response": "No emissions could be calculated",
                }

            # Aggregate emissions
            agg_result = client.aggregate_emissions(emissions_list)

            # Format response
            response_parts = []
            if agg_result["success"]:
                carbon_data = agg_result["data"]
                total_emissions = carbon_data.get("total_co2e_tons", 0)

                response_parts.append("Carbon Footprint Analysis")
                if country != "US":
                    response_parts.append(f"Country/Region: {country}")
                response_parts.append(f"{'=' * 40}")
                response_parts.append(
                    f"Total Emissions: {total_emissions:.3f} metric tons CO2e"
                )
                response_parts.append(
                    f"Total Emissions: {carbon_data.get('total_co2e_kg', 0):.2f} kg CO2e"
                )

                if "emissions_breakdown" in carbon_data:
                    response_parts.append("\nEmissions Breakdown:")
                    for item in carbon_data["emissions_breakdown"]:
                        response_parts.append(
                            f"  - {item['source']}: {item['co2e_tons']:.3f} tons ({item['percentage']}%)"
                        )

            response = (
                "\n".join(response_parts)
                if response_parts
                else "No emissions data calculated"
            )

            return {"success": True, "response": response}

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"Error processing request: {str(e)}",
            }

    def _create_dynamic_workflow(self, data: Dict[str, Any]):
        builder = WorkflowBuilder(
            "dynamic_carbon_calculation", "Dynamic workflow for carbon calculation"
        )

        # Only add validator if we have the proper data structure
        if "fuels" in data and isinstance(data["fuels"], list):
            builder.add_step("validate", "validator")

        for i, fuel in enumerate(data.get("fuels", [])):
            builder.add_step(f"calculate_fuel_{i}", "fuel")

        builder.add_step("aggregate", "carbon")

        if data.get("building_info", {}).get("area"):
            builder.add_step("benchmark", "benchmark")

        builder.add_step("report", "report")

        return builder.build()

    def _format_response(
        self, result: Dict[str, Any], input_data: Dict[str, Any]
    ) -> str:
        if not result["success"]:
            return f"Error calculating emissions: {result.get('errors', ['Unknown error'])[0]}"

        response_parts = []

        if "results" in result:
            if "aggregate" in result["results"]:
                carbon_data = result["results"]["aggregate"].data
                total_emissions = carbon_data.get("total_co2e_tons", 0)

                response_parts.append("Carbon Footprint Analysis\n")
                response_parts.append(f"{'=' * 40}\n")
                response_parts.append(
                    f"Total Emissions: {total_emissions:.3f} metric tons CO2e\n"
                )

                if "emissions_breakdown" in carbon_data:
                    response_parts.append("\nEmissions Breakdown:")
                    for item in carbon_data["emissions_breakdown"]:
                        response_parts.append(
                            f"  - {item['source']}: {item['co2e_tons']:.3f} tons ({item['percentage']}%)"
                        )

                if (
                    "carbon_intensity" in carbon_data
                    and carbon_data["carbon_intensity"]
                ):
                    response_parts.append("\nCarbon Intensity:")
                    for key, value in carbon_data["carbon_intensity"].items():
                        unit = key.replace("per_", "").replace("_", " ")
                        response_parts.append(f"  - {value:.2f} kg CO2e per {unit}")

            if "benchmark" in result["results"]:
                benchmark_data = result["results"]["benchmark"].data
                response_parts.append("\nBenchmark Comparison:")
                response_parts.append(f"  - Rating: {benchmark_data['rating']}")
                response_parts.append(
                    f"  - Percentile: Top {benchmark_data['percentile']}%"
                )

                if "recommendations" in benchmark_data:
                    response_parts.append("\nRecommendations:")
                    for rec in benchmark_data["recommendations"][:3]:
                        response_parts.append(f"  - {rec}")

            if "report" in result["results"]:
                report_data = result["results"]["report"].data
                if "report" in report_data and isinstance(report_data["report"], str):
                    if len(response_parts) == 0:
                        response_parts.append(report_data["report"])

        return (
            "\n".join(response_parts)
            if response_parts
            else "No emissions data calculated"
        )

    def _create_llm_prompt(self, query: str) -> str:
        return f"""Extract carbon emission data from this query: "{query}"

Return a JSON object with:
- fuels: array of {{fuel_type, consumption, unit}}
- building_info: {{type, area, area_unit, occupancy}}

Fuel types: electricity, natural_gas, diesel, gasoline, propane
Electricity units: kWh, MWh, GWh
Natural gas units: therms, ccf, m3
Liquid fuel units: gallons, liters

If the query is not about calculating emissions, return an empty JSON object.

Example:
{{
  "fuels": [
    {{"fuel_type": "electricity", "consumption": 1000, "unit": "kWh"}},
    {{"fuel_type": "natural_gas", "consumption": 500, "unit": "therms"}}
  ],
  "building_info": {{
    "type": "commercial_office",
    "area": 10000,
    "area_unit": "sqft",
    "occupancy": 50
  }}
}}"""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        try:
            import json

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except:
            return {}
