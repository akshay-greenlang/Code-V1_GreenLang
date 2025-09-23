"""GridFactorAgent with full type hints."""

from typing import Dict, List
import json
from pathlib import Path
import logging

from ..types import Agent, AgentResult, ErrorInfo, CountryCode
from .types import GridFactorInput, GridFactorOutput

logger = logging.getLogger(__name__)


class GridFactorAgent(Agent[GridFactorInput, GridFactorOutput]):
    """Agent for retrieving country/region-specific emission factors."""

    agent_id: str = "grid_factor"
    name: str = "Grid Emission Factor Provider"
    version: str = "0.0.1"

    def __init__(self) -> None:
        """Initialize with emission factors database."""
        self.factors_path = (
            Path(__file__).parent.parent / "data" / "global_emission_factors.json"
        )
        self.emission_factors: Dict[str, Dict[str, Dict[str, float]]] = (
            self._load_emission_factors()
        )

    def _load_emission_factors(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Load emission factors from JSON file."""
        try:
            with open(self.factors_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load emission factors: {e}")
            # Fallback to basic factors
            return {
                "US": {
                    "electricity": {"emission_factor": 0.385, "unit": "kgCO2e/kWh"},
                    "natural_gas": {"emission_factor": 5.3, "unit": "kgCO2e/therm"},
                    "diesel": {"emission_factor": 10.21, "unit": "kgCO2e/gallon"},
                }
            }

    def validate(self, payload: GridFactorInput) -> bool:
        """Validate input payload."""
        if (
            not payload.get("country")
            or not payload.get("fuel_type")
            or not payload.get("unit")
        ):
            return False
        return True

    def run(self, payload: GridFactorInput) -> AgentResult[GridFactorOutput]:
        """Get emission factor for specific country, fuel type, and unit."""
        if not self.validate(payload):
            error_info: ErrorInfo = {
                "type": "ValidationError",
                "message": "Missing required fields: country, fuel_type, unit",
                "agent_id": self.agent_id,
                "context": {"payload": payload},
            }
            return {"success": False, "error": error_info}

        try:
            country = payload["country"]
            fuel_type = payload["fuel_type"]
            unit = payload["unit"]
            year = payload.get("year", 2025)

            # Map country codes if needed
            country_mapping: Dict[str, CountryCode] = {
                "USA": "US",
                "INDIA": "IN",
                "CHINA": "CN",
                "JAPAN": "JP",
                "BRAZIL": "BR",
                "SOUTH_KOREA": "KR",
                "KOREA": "KR",
                "GERMANY": "DE",
                "FRANCE": "FR",
                "CANADA": "CA",
                "AUSTRALIA": "AU",
                "UNITED_KINGDOM": "UK",
            }

            mapped_country = country_mapping.get(country.upper(), country)

            # Check if country exists
            if mapped_country not in self.emission_factors:
                logger.warning(f"Country {mapped_country} not found, using US factors")
                mapped_country = "US"

            country_factors = self.emission_factors[mapped_country]

            # Get fuel type factors
            if fuel_type not in country_factors:
                error_info: ErrorInfo = {
                    "type": "DataError",
                    "message": f"Fuel type '{fuel_type}' not found for country {mapped_country}",
                    "agent_id": self.agent_id,
                    "context": {"fuel_type": fuel_type, "country": mapped_country},
                }
                return {"success": False, "error": error_info}

            fuel_data = country_factors[fuel_type]

            # Get emission factor based on unit
            # Check if fuel_data has the unit as a key (new structure)
            if isinstance(fuel_data, dict) and unit in fuel_data:
                factor = fuel_data[unit]
            # Fallback to old structure with emission_factor key
            elif isinstance(fuel_data, dict) and "emission_factor" in fuel_data:
                factor = fuel_data["emission_factor"]
            else:
                factor = 0.0

            if factor == 0.0:
                error_info: ErrorInfo = {
                    "type": "DataError",
                    "message": f"No emission factor found for {fuel_type} with unit {unit} in {mapped_country}",
                    "agent_id": self.agent_id,
                    "context": {
                        "fuel_type": fuel_type,
                        "country": mapped_country,
                        "unit": unit,
                        "available_units": (
                            list(fuel_data.keys())
                            if isinstance(fuel_data, dict)
                            else []
                        ),
                    },
                }
                return {"success": False, "error": error_info}

            # Build output
            output: GridFactorOutput = {
                "emission_factor": factor,
                "unit": f"kgCO2e/{unit}",
                "source": (
                    fuel_data.get(
                        "source",
                        fuel_data.get("description", "GreenLang Global Dataset"),
                    )
                    if isinstance(fuel_data, dict)
                    else "GreenLang Global Dataset"
                ),
                "version": (
                    fuel_data.get("version", "1.0.0")
                    if isinstance(fuel_data, dict)
                    else "1.0.0"
                ),
                "last_updated": (
                    fuel_data.get("last_updated", "2025-08-14")
                    if isinstance(fuel_data, dict)
                    else "2025-08-14"
                ),
                "country": mapped_country,
                "fuel_type": fuel_type,
            }

            # Add grid mix if available
            if "grid_renewable_share" in country_factors:
                output["grid_mix"] = {
                    "renewable": country_factors["grid_renewable_share"],
                    "fossil": 1.0 - country_factors["grid_renewable_share"],
                }

            return {
                "success": True,
                "data": output,
                "metadata": {
                    "agent_id": self.agent_id,
                    "source": self.emission_factors.get("metadata", {}).get(
                        "source", "IPCC"
                    ),
                    "methodology": self.emission_factors.get("metadata", {}).get(
                        "methodology", "IPCC Guidelines"
                    ),
                },
            }

        except Exception as e:
            error_info: ErrorInfo = {
                "type": "CalculationError",
                "message": f"Error retrieving emission factor: {str(e)}",
                "agent_id": self.agent_id,
                "traceback": str(e),
            }
            return {"success": False, "error": error_info}

    def get_available_countries(self) -> List[str]:
        """Get list of available countries."""
        return [k for k in self.emission_factors.keys() if k != "metadata"]

    def get_available_fuel_types(self, country: str) -> List[str]:
        """Get available fuel types for a country."""
        country = country.upper()
        if country in self.emission_factors:
            return [
                k
                for k in self.emission_factors[country].keys()
                if not k.startswith("grid_renewable")
            ]
        return []
