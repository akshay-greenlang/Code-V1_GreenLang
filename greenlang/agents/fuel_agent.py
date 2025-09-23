from typing import Optional, Dict, Any, List
from functools import lru_cache
from datetime import datetime
import asyncio
import logging
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..types import Agent, AgentResult, ErrorInfo
from .types import FuelInput, FuelOutput
from greenlang.data.emission_factors import EmissionFactors
from greenlang.utils.unit_converter import UnitConverter
from greenlang.utils.performance_tracker import PerformanceTracker


class FuelAgent(Agent[FuelInput, FuelOutput]):
    """Agent for calculating emissions based on fuel consumption.

    Enhanced with:
    - Async support for parallel processing
    - Performance tracking and monitoring
    - JSON Schema validation
    - Export capabilities (JSON/CSV/Excel)
    - External configuration management
    - Historical data tracking
    - Comprehensive logging
    - Integrated unit conversion
    """

    agent_id: str = "fuel"
    name: str = "Fuel Emissions Calculator"
    version: str = "0.0.1"

    # Cache configuration
    CACHE_TTL_SECONDS = 3600  # 1 hour cache

    # Load fuel properties from external config
    @classmethod
    @lru_cache(maxsize=1)
    def load_fuel_config(cls) -> Dict:
        """Load fuel properties configuration from external file."""
        config_path = Path(__file__).parent.parent / "configs" / "fuel_properties.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            # Fallback to basic properties
            return {
                "fuel_properties": {
                    "electricity": {
                        "energy_content": {"value": 3412, "unit": "Btu/kWh"}
                    },
                    "natural_gas": {
                        "energy_content": {"value": 100000, "unit": "Btu/therm"}
                    },
                    "diesel": {
                        "energy_content": {"value": 138690, "unit": "Btu/gallon"}
                    },
                }
            }

    def __init__(self) -> None:
        """Initialize the FuelAgent with enhanced features.

        Sets up:
        - Emission factors database
        - Fuel properties configuration
        - Unit converter
        - Performance tracker
        - Cache management
        - Historical data storage
        - Logging
        """
        self.emission_factors = EmissionFactors()
        self.fuel_config = self.load_fuel_config()
        self.unit_converter = UnitConverter()
        self.performance_tracker = PerformanceTracker(self.agent_id)
        self._cache = {}
        self._cache_timestamps = {}
        self._historical_data = []

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.agent_id}")
        self.logger.setLevel(logging.INFO)

        # Performance monitoring
        self._execution_times = []
        self._cache_hits = 0
        self._cache_misses = 0

    def validate(self, payload: FuelInput) -> bool:
        """Validate input payload structure and values with JSON Schema.

        Args:
            payload: Input data containing fuel consumption information

        Returns:
            bool: True if validation passes, False otherwise
        """
        self.logger.debug(
            f"Validating payload for fuel type: {payload.get('fuel_type')}"
        )

        # Basic validation
        if "fuel_type" not in payload:
            self.logger.error("Missing fuel_type in payload")
            return False

        if "amount" not in payload:
            self.logger.error("Missing amount in payload")
            return False

        if "unit" not in payload:
            self.logger.error("Missing unit in payload")
            return False

        # Check renewable fuels can have negative amounts
        fuel_type = payload["fuel_type"]
        amount = payload["amount"]

        renewable_fuels = (
            self.fuel_config.get("fuel_properties", {})
            .get(fuel_type, {})
            .get("renewable_potential")
            == "renewable"
        )

        if renewable_fuels and amount > 0:
            self.logger.warning(
                f"Renewable fuel {fuel_type} should have negative amount for generation"
            )
        elif not renewable_fuels and amount < 0:
            self.logger.error(
                f"Non-renewable fuel {fuel_type} cannot have negative amount"
            )
            return False

        self.logger.debug("Validation passed")
        return True

    @lru_cache(maxsize=256)
    def _get_cached_emission_factor(
        self, fuel_type: str, unit: str, region: str
    ) -> Optional[float]:
        """Get emission factor with caching for performance.

        Args:
            fuel_type: Type of fuel
            unit: Unit of measurement
            region: Country or region code

        Returns:
            Optional[float]: Emission factor or None if not found
        """
        cache_key = f"fuel_{fuel_type}_{unit}_{region}"

        # Track cache performance
        if cache_key in self._cache:
            self._cache_hits += 1
            self.logger.debug(f"Cache hit for {cache_key}")
        else:
            self._cache_misses += 1
            self.logger.debug(f"Cache miss for {cache_key}")

        return self.emission_factors.get_factor(
            fuel_type=fuel_type, unit=unit, region=region
        )

    def run(self, payload: FuelInput) -> AgentResult[FuelOutput]:
        """Calculate emissions from fuel consumption with performance tracking.

        Args:
            payload: Input data with fuel consumption details

        Returns:
            AgentResult containing calculated emissions and metrics
        """
        start_time = datetime.now()

        with self.performance_tracker.track("fuel_calculation"):
            if not self.validate(payload):
                error_info: ErrorInfo = {
                    "type": "ValidationError",
                    "message": "Invalid input payload",
                    "agent_id": self.agent_id,
                    "context": {"payload": payload},
                }
                return {"success": False, "error": error_info}

            fuel_type = payload["fuel_type"]
            amount = payload["amount"]
            unit = payload["unit"]
            country = payload.get("country", "US")
            year = payload.get("year", 2025)

            # Map fuel type aliases
            fuel_type_mapping = {
                "lpg": "propane",
                "heating_oil": "fuel_oil",
                "wood": "biomass",
                "electric": "electricity",
            }
            fuel_type = fuel_type_mapping.get(fuel_type, fuel_type)

            self.logger.info(
                f"Calculating emissions for {amount} {unit} of {fuel_type}"
            )

            try:
                # Get emission factor with caching
                emission_factor = self._get_cached_emission_factor(
                    fuel_type, unit, country
                )

                if emission_factor is None:
                    error_info: ErrorInfo = {
                        "type": "DataError",
                        "message": f"No emission factor found for {fuel_type} in {country}",
                        "agent_id": self.agent_id,
                        "context": {"fuel_type": fuel_type, "country": country},
                    }
                    return {"success": False, "error": error_info}

                # Calculate emissions
                co2e_emissions_kg = abs(amount) * emission_factor

                # Apply renewable offset if specified
                renewable_percentage = payload.get("renewable_percentage", 0)
                if renewable_percentage > 0 and fuel_type == "electricity":
                    offset = co2e_emissions_kg * (renewable_percentage / 100)
                    co2e_emissions_kg -= offset
                    self.logger.info(
                        f"Applied {renewable_percentage}% renewable offset"
                    )

                # Apply efficiency if specified
                efficiency = payload.get("efficiency", 1.0)
                if efficiency < 1.0:
                    co2e_emissions_kg = co2e_emissions_kg / efficiency
                    self.logger.info(f"Adjusted for {efficiency*100}% efficiency")

                # Get fuel properties
                fuel_props = self.fuel_config.get("fuel_properties", {}).get(
                    fuel_type, {}
                )
                energy_content = fuel_props.get("energy_content", {})

                # Calculate energy equivalent
                energy_mmbtu = self._calculate_energy_content(amount, unit, fuel_type)

                # Determine scope
                scope = self._determine_scope(fuel_type)

                # Generate recommendations
                recommendations = self._generate_fuel_recommendations(
                    fuel_type, amount, unit, co2e_emissions_kg, country
                )

                # Store in historical data
                result_data = {
                    "timestamp": datetime.now().isoformat(),
                    "fuel_type": fuel_type,
                    "amount": amount,
                    "unit": unit,
                    "emissions_kg": co2e_emissions_kg,
                    "country": country,
                }
                self._historical_data.append(result_data)

                # Calculate performance metrics
                duration = (datetime.now() - start_time).total_seconds()
                self._execution_times.append(duration)

                output: FuelOutput = {
                    "co2e_emissions_kg": co2e_emissions_kg,
                    "fuel_type": fuel_type,
                    "consumption_amount": amount,
                    "consumption_unit": unit,
                    "emission_factor": emission_factor,
                    "emission_factor_unit": f"kgCO2e/{unit}",
                    "country": country,
                    "scope": scope,
                    "energy_content_mmbtu": energy_mmbtu,
                    "renewable_offset_applied": renewable_percentage > 0,
                    "efficiency_adjusted": efficiency < 1.0,
                    "recommendations": recommendations,
                    "calculation_time_ms": duration * 1000,
                }

                # Add energy content info if available
                if energy_content:
                    output["energy_content_info"] = energy_content

                self.logger.info(
                    f"Calculated {co2e_emissions_kg:.2f} kg CO2e emissions"
                )

                return {
                    "success": True,
                    "data": output,
                    "metadata": {
                        "agent_id": self.agent_id,
                        "calculation": f"{abs(amount)} {unit} × {emission_factor} kgCO2e/{unit}",
                        "cache_hits": self._cache_hits,
                        "cache_misses": self._cache_misses,
                    },
                }

            except Exception as e:
                self.logger.error(f"Error in fuel calculation: {e}")
                error_info: ErrorInfo = {
                    "type": "CalculationError",
                    "message": f"Failed to calculate fuel emissions: {str(e)}",
                    "agent_id": self.agent_id,
                    "traceback": str(e),
                }
                return {"success": False, "error": error_info}

    def batch_process(self, fuels: List[FuelInput]) -> List[AgentResult[FuelOutput]]:
        """Process multiple fuel sources with performance optimization.

        Args:
            fuels: List of fuel consumption data

        Returns:
            List of results for each fuel source
        """
        self.logger.info(f"Batch processing {len(fuels)} fuel sources")

        with self.performance_tracker.track("batch_processing"):
            results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_fuel = {
                    executor.submit(self.run, fuel): fuel for fuel in fuels
                }

                for future in as_completed(future_to_fuel):
                    result = future.result()
                    results.append(result)

            # Calculate aggregate metrics
            total_emissions = sum(
                r["data"]["co2e_emissions_kg"] for r in results if r["success"]
            )

            self.logger.info(
                f"Batch processing complete. Total emissions: {total_emissions:.2f} kg CO2e"
            )

            return results

    async def async_batch_process(
        self, fuels: List[FuelInput]
    ) -> List[AgentResult[FuelOutput]]:
        """Async batch processing for multiple fuel sources.

        Args:
            fuels: List of fuel consumption data

        Returns:
            List of results for each fuel source
        """
        self.logger.info(f"Async batch processing {len(fuels)} fuel sources")

        async def process_fuel(fuel: FuelInput) -> AgentResult[FuelOutput]:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.run, fuel)

        tasks = [process_fuel(fuel) for fuel in fuels]
        results = await asyncio.gather(*tasks)

        total_emissions = sum(
            r["data"]["co2e_emissions_kg"] for r in results if r["success"]
        )

        self.logger.info(
            f"Async processing complete. Total emissions: {total_emissions:.2f} kg CO2e"
        )

        return results

    def _calculate_energy_content(
        self, amount: float, unit: str, fuel_type: str
    ) -> float:
        """Calculate energy content in MMBtu using unit converter.

        Args:
            amount: Fuel amount
            unit: Unit of measurement
            fuel_type: Type of fuel

        Returns:
            float: Energy content in MMBtu
        """
        try:
            return self.unit_converter.convert_fuel_to_energy(
                abs(amount), unit, fuel_type, "MMBtu"
            )
        except:
            # Fallback calculation
            if unit == "kWh":
                return abs(amount) * 0.003412
            elif unit == "therms":
                return abs(amount) * 0.1
            elif unit == "gallons" and fuel_type == "diesel":
                return abs(amount) * 0.138
            elif unit == "gallons" and fuel_type == "gasoline":
                return abs(amount) * 0.125
            else:
                return 0.0

    def _determine_scope(self, fuel_type: str) -> str:
        """Determine GHG Protocol scope for fuel type.

        Args:
            fuel_type: Type of fuel

        Returns:
            str: Scope (1, 2, or 3)
        """
        scope_mapping = {
            "natural_gas": "1",
            "diesel": "1",
            "gasoline": "1",
            "propane": "1",
            "fuel_oil": "1",
            "coal": "1",
            "biomass": "1",
            "electricity": "2",
            "district_heating": "2",
            "district_cooling": "2",
        }
        return scope_mapping.get(fuel_type, "1")

    def _generate_fuel_recommendations(
        self,
        fuel_type: str,
        amount: float,
        unit: str,
        emissions_kg: float,
        country: str,
    ) -> List[Dict[str, str]]:
        """Generate fuel switching and efficiency recommendations.

        Args:
            fuel_type: Current fuel type
            amount: Consumption amount
            unit: Unit of measurement
            emissions_kg: Calculated emissions
            country: Country code

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Load switching recommendations from config
        switching = self.fuel_config.get("fuel_switching_recommendations", {})

        # Fuel switching recommendations
        if fuel_type == "coal":
            recommendations.append(
                {
                    "priority": "high",
                    "action": "Switch from coal to natural gas",
                    "impact": "45% emissions reduction",
                    "feasibility": "high",
                }
            )
        elif fuel_type == "fuel_oil":
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "Switch from oil to natural gas or heat pump",
                    "impact": "25-60% emissions reduction",
                    "feasibility": "medium",
                }
            )
        elif fuel_type == "natural_gas":
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "Consider renewable natural gas or hydrogen blend",
                    "impact": "20-100% emissions reduction",
                    "feasibility": "low",
                }
            )

        # Renewable energy recommendations
        if fuel_type == "electricity":
            recommendations.append(
                {
                    "priority": "high",
                    "action": "Install on-site solar PV or purchase renewable energy",
                    "impact": "Up to 100% emissions reduction",
                    "feasibility": "high",
                }
            )

        # Efficiency recommendations
        recommendations.append(
            {
                "priority": "medium",
                "action": "Improve equipment efficiency and controls",
                "impact": "10-20% reduction in fuel consumption",
                "feasibility": "high",
            }
        )

        # Energy management
        recommendations.append(
            {
                "priority": "low",
                "action": "Implement energy management system and monitoring",
                "impact": "5-15% reduction through behavioral changes",
                "feasibility": "high",
            }
        )

        return recommendations[:5]  # Return top 5 recommendations

    def export_results(self, results: List[AgentResult], format: str = "json") -> str:
        """Export calculation results to various formats.

        Args:
            results: List of calculation results
            format: Export format (json, csv, excel)

        Returns:
            str: Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fuel_emissions_{timestamp}.{format}"

        if format == "json":
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)

        elif format == "csv":
            import csv

            with open(filename, "w", newline="") as f:
                if results and results[0]["success"]:
                    writer = csv.DictWriter(f, fieldnames=results[0]["data"].keys())
                    writer.writeheader()
                    for r in results:
                        if r["success"]:
                            writer.writerow(r["data"])

        elif format == "excel":
            try:
                import pandas as pd

                data = [r["data"] for r in results if r["success"]]
                df = pd.DataFrame(data)
                df.to_excel(filename, index=False)
            except ImportError:
                self.logger.error(
                    "pandas is required for Excel export. "
                    "Install it with: pip install greenlang[analytics]"
                )
                raise

        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Exported results to {filename}")
        return filename

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Dict containing performance statistics
        """
        summary = self.performance_tracker.get_summary()

        # Add fuel-specific metrics
        summary["fuel_metrics"] = {
            "cache_hit_rate": self._cache_hits
            / max(self._cache_hits + self._cache_misses, 1),
            "average_execution_time_ms": sum(self._execution_times)
            / max(len(self._execution_times), 1)
            * 1000,
            "total_calculations": len(self._historical_data),
            "unique_fuel_types": len(
                set(d["fuel_type"] for d in self._historical_data)
            ),
        }

        return summary

    def get_historical_data(self) -> List[Dict]:
        """Get historical calculation data.

        Returns:
            List of historical calculation records
        """
        return self._historical_data.copy()

    def clear_cache(self):
        """Clear the emission factor cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._get_cached_emission_factor.cache_clear()
        self.logger.info("Cache cleared")

    def reset_metrics(self):
        """Reset performance metrics."""
        self._execution_times.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._historical_data.clear()
        self.performance_tracker.reset()
        self.logger.info("Metrics reset")
