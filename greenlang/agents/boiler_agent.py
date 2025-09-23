from typing import Optional, Dict, Any, List
from functools import lru_cache
from datetime import datetime
import asyncio
import logging
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..types import Agent, AgentResult, ErrorInfo
from .types import BoilerInput, BoilerOutput
from greenlang.data.emission_factors import EmissionFactors
from greenlang.utils.unit_converter import UnitConverter
from greenlang.utils.performance_tracker import PerformanceTracker


class BoilerAgent(Agent[BoilerInput, BoilerOutput]):
    """Agent for calculating emissions from boiler operations and thermal systems.

    This agent specializes in:
    - Boiler efficiency calculations with caching
    - Thermal output to fuel consumption conversion
    - Multi-fuel boiler systems with batch processing
    - Steam and hot water generation emissions
    - Boiler performance optimization recommendations
    - Async support for parallel processing
    - Historical performance tracking
    - Export to multiple formats
    """

    agent_id: str = "boiler"
    name: str = "Boiler Emissions Calculator"
    version: str = "0.0.1"

    # Cache configuration
    CACHE_TTL_SECONDS = 3600  # 1 hour cache

    # Load efficiency data from external config
    @classmethod
    @lru_cache(maxsize=1)
    def load_efficiency_config(cls) -> Dict:
        """Load boiler efficiency configuration from external file."""
        config_path = (
            Path(__file__).parent.parent / "configs" / "boiler_efficiencies.json"
        )
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            # Fallback to hardcoded values
            return {
                "natural_gas": {
                    "condensing": {"new": 0.95, "medium": 0.92, "old": 0.88},
                    "standard": {"new": 0.85, "medium": 0.80, "old": 0.75},
                    "low_efficiency": {"new": 0.78, "medium": 0.72, "old": 0.65},
                },
                "oil": {
                    "condensing": {"new": 0.92, "medium": 0.88, "old": 0.84},
                    "standard": {"new": 0.83, "medium": 0.78, "old": 0.72},
                    "low_efficiency": {"new": 0.75, "medium": 0.68, "old": 0.60},
                },
                "fuel_oil": {
                    "condensing": {"new": 0.92, "medium": 0.88, "old": 0.84},
                    "standard": {"new": 0.83, "medium": 0.78, "old": 0.72},
                    "low_efficiency": {"new": 0.75, "medium": 0.68, "old": 0.60},
                },
                "biomass": {
                    "modern": {"new": 0.85, "medium": 0.80, "old": 0.75},
                    "standard": {"new": 0.75, "medium": 0.70, "old": 0.65},
                    "traditional": {"new": 0.65, "medium": 0.55, "old": 0.45},
                },
                "coal": {
                    "pulverized": {"new": 0.85, "medium": 0.80, "old": 0.75},
                    "stoker": {"new": 0.75, "medium": 0.70, "old": 0.65},
                    "hand_fired": {"new": 0.60, "medium": 0.50, "old": 0.40},
                },
                "electric": {
                    "resistance": {"new": 0.99, "medium": 0.98, "old": 0.97},
                    "heat_pump": {
                        "new": 3.50,
                        "medium": 3.00,
                        "old": 2.50,
                    },  # COP values
                },
            }

    def __init__(self) -> None:
        """Initialize the BoilerAgent with emission factors database and performance tracking.

        Sets up:
        - Emission factors database
        - Efficiency configuration
        - Unit converter
        - Performance tracker
        - Cache management
        - Logging
        """
        self.emission_factors = EmissionFactors()
        self.BOILER_EFFICIENCIES = self.load_efficiency_config()
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

    def validate(self, payload: BoilerInput) -> bool:
        """Validate input payload structure and values.

        Args:
            payload: Input data containing boiler specifications

        Returns:
            bool: True if validation passes, False otherwise
        """
        self.logger.debug(
            f"Validating payload for boiler type: {payload.get('boiler_type')}"
        )

        # Check required fields
        if not payload.get("boiler_type"):
            self.logger.error("Missing boiler_type in payload")
            return False

        # Validate based on input type (thermal output or fuel consumption)
        if "thermal_output" in payload:
            thermal = payload["thermal_output"]
            if not isinstance(thermal, dict):
                return False
            if "value" not in thermal or "unit" not in thermal:
                return False
            if thermal["value"] <= 0:
                return False
        elif "fuel_consumption" in payload:
            fuel = payload["fuel_consumption"]
            if not isinstance(fuel, dict):
                return False
            if "value" not in fuel or "unit" not in fuel:
                return False
            if fuel["value"] <= 0:
                return False
        else:
            # Must have either thermal output or fuel consumption
            self.logger.error("Neither thermal_output nor fuel_consumption provided")
            return False

        # Validate efficiency if provided
        if "efficiency" in payload:
            eff = payload["efficiency"]
            if not isinstance(eff, (int, float)):
                return False
            # Efficiency should be between 0 and 1 (or 0-100 for percentage)
            if eff <= 0:
                return False
            # Convert percentage to decimal if needed
            if eff > 1 and eff <= 100:
                payload["efficiency"] = eff / 100
            elif eff > 100:
                return False

        self.logger.debug("Validation passed")
        return True

    @lru_cache(maxsize=256)
    def _get_cached_emission_factor(
        self, fuel_type: str, unit: str, region: str, year: int
    ) -> Optional[Dict]:
        """Get emission factor with caching for performance.

        Args:
            fuel_type: Type of fuel
            unit: Unit of measurement
            region: Country or region code
            year: Year for historical factors

        Returns:
            Optional[Dict]: Emission factor info or None if not found
        """
        cache_key = f"boiler_{fuel_type}_{unit}_{region}_{year}"

        # Track cache performance
        if cache_key in self._cache:
            self._cache_hits += 1
            self.logger.debug(f"Cache hit for {cache_key}")
        else:
            self._cache_misses += 1
            self.logger.debug(f"Cache miss for {cache_key}")

        # Get the emission factor
        factor = self.emission_factors.get_factor(
            fuel_type=fuel_type, unit=unit, region=region
        )

        if factor is None:
            return None

        # Return with metadata structure
        return {
            "emission_factor": factor,
            "source": "GreenLang Global Dataset",
            "version": "1.0.0",
            "last_updated": "2025-08-14",
            "confidence": "high",
        }

    def run(self, payload: BoilerInput) -> AgentResult[BoilerOutput]:
        """Calculate emissions from boiler operations with performance tracking.

        Args:
            payload: Input data with boiler specifications

        Returns:
            AgentResult containing calculated emissions and boiler metrics
        """
        start_time = datetime.now()

        with self.performance_tracker.track("boiler_calculation"):
            if not self.validate(payload):
                error_info: ErrorInfo = {
                    "type": "ValidationError",
                    "message": "Invalid input payload for boiler calculations",
                    "agent_id": self.agent_id,
                    "context": {"payload": payload},
                }
                return {"success": False, "error": error_info}

            boiler_type = payload["boiler_type"]
            fuel_type = payload.get("fuel_type", "natural_gas")

            # Map common fuel type aliases
            fuel_type_mapping = {
                "oil": "fuel_oil",
                "heating_oil": "fuel_oil",
                "diesel_oil": "diesel",
                "lpg": "propane",
                "wood": "biomass",
                "electric": "electricity",
            }
            fuel_type = fuel_type_mapping.get(fuel_type, fuel_type)

            country = payload.get("country", "US")
            year = payload.get("year", 2025)

            # Determine boiler efficiency
            efficiency = self._get_efficiency(payload, boiler_type, fuel_type)

            try:
                # Calculate fuel consumption based on input type
                if "thermal_output" in payload:
                    fuel_consumption = self._calculate_fuel_from_thermal(
                        payload["thermal_output"], efficiency, fuel_type
                    )
                else:
                    fuel_consumption = payload["fuel_consumption"]

                # Get emission factor with caching
                factor_info = self._get_cached_emission_factor(
                    fuel_type=fuel_type,
                    unit=fuel_consumption["unit"],
                    region=country,
                    year=year,
                )

                if factor_info is None:
                    error_info: ErrorInfo = {
                        "type": "DataError",
                        "message": f"No emission factor found for {fuel_type} in {country}",
                        "agent_id": self.agent_id,
                        "context": {"fuel_type": fuel_type, "country": country},
                    }
                    return {"success": False, "error": error_info}

                emission_factor = factor_info["emission_factor"]
                co2e_emissions_kg = fuel_consumption["value"] * emission_factor

                # Calculate thermal output if not provided
                if "thermal_output" in payload:
                    thermal_output = payload["thermal_output"]
                else:
                    thermal_output = self._calculate_thermal_from_fuel(
                        fuel_consumption, efficiency, fuel_type
                    )

                # Calculate performance metrics
                thermal_efficiency_percent = efficiency * 100
                fuel_intensity = (
                    fuel_consumption["value"] / thermal_output["value"]
                    if thermal_output["value"] > 0
                    else 0
                )
                emission_intensity = (
                    co2e_emissions_kg / thermal_output["value"]
                    if thermal_output["value"] > 0
                    else 0
                )

                # Generate optimization recommendations
                recommendations = self._generate_recommendations(
                    boiler_type, fuel_type, efficiency, payload.get("age", "medium")
                )

                # Track historical data
                self._track_historical_data(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "boiler_type": boiler_type,
                        "efficiency": efficiency,
                        "emissions_kg": co2e_emissions_kg,
                        "fuel_consumption": fuel_consumption["value"],
                    }
                )

                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds()
                self._execution_times.append(execution_time)

                output: BoilerOutput = {
                    "co2e_emissions_kg": co2e_emissions_kg,
                    "boiler_type": boiler_type,
                    "fuel_type": fuel_type,
                    "fuel_consumption_value": fuel_consumption["value"],
                    "fuel_consumption_unit": fuel_consumption["unit"],
                    "thermal_output_value": thermal_output["value"],
                    "thermal_output_unit": thermal_output["unit"],
                    "efficiency": efficiency,
                    "thermal_efficiency_percent": thermal_efficiency_percent,
                    "emission_factor": emission_factor,
                    "emission_factor_unit": f"kgCO2e/{fuel_consumption['unit']}",
                    "fuel_intensity": fuel_intensity,
                    "emission_intensity": emission_intensity,
                    "recommendations": recommendations,
                    "source": factor_info.get("source", "GreenLang Global Dataset"),
                    "version": factor_info.get("version", "1.0.0"),
                    "last_updated": factor_info.get("last_updated", "2025-08-14"),
                }

                if "confidence" in factor_info:
                    output["confidence"] = factor_info["confidence"]

                # Add performance rating
                output["performance_rating"] = self._get_performance_rating(
                    efficiency, boiler_type, fuel_type
                )

                # Log performance metrics
                self.logger.info(
                    f"Boiler calculation completed in {execution_time:.3f}s"
                )

                return {
                    "success": True,
                    "data": output,
                    "metadata": {
                        "agent_id": self.agent_id,
                        "calculation": f"Fuel: {fuel_consumption['value']} {fuel_consumption['unit']} × {emission_factor} kgCO2e/{fuel_consumption['unit']}",
                        "efficiency_used": f"{thermal_efficiency_percent:.1f}%",
                        "thermal_output": f"{thermal_output['value']} {thermal_output['unit']}",
                        "execution_time_ms": execution_time * 1000,
                        "cache_hit_rate": self._get_cache_hit_rate(),
                    },
                }

            except Exception as e:
                self.logger.error(
                    f"Error in boiler calculation: {str(e)}", exc_info=True
                )
                error_info: ErrorInfo = {
                    "type": "CalculationError",
                    "message": f"Failed to calculate boiler emissions: {str(e)}",
                    "agent_id": self.agent_id,
                    "traceback": str(e),
                }
                return {"success": False, "error": error_info}

    def batch_process(
        self, boilers: List[BoilerInput]
    ) -> List[AgentResult[BoilerOutput]]:
        """Process multiple boilers in batch for performance.

        Args:
            boilers: List of boiler input data

        Returns:
            List of AgentResult for each boiler
        """
        self.logger.info(f"Batch processing {len(boilers)} boilers")
        results = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.run, boiler): i for i, boiler in enumerate(boilers)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    self.logger.error(f"Error processing boiler {idx}: {str(e)}")
                    error_result = {
                        "success": False,
                        "error": {
                            "type": "BatchProcessingError",
                            "message": f"Failed to process boiler {idx}: {str(e)}",
                            "agent_id": self.agent_id,
                        },
                    }
                    results.append((idx, error_result))

        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    async def async_run(self, payload: BoilerInput) -> AgentResult[BoilerOutput]:
        """Async version of run method for concurrent processing.

        Args:
            payload: Input data with boiler specifications

        Returns:
            AgentResult containing calculated emissions
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, payload)

    async def async_batch_process(
        self, boilers: List[BoilerInput]
    ) -> List[AgentResult[BoilerOutput]]:
        """Process multiple boilers asynchronously.

        Args:
            boilers: List of boiler input data

        Returns:
            List of AgentResult for each boiler
        """
        self.logger.info(f"Async batch processing {len(boilers)} boilers")
        tasks = [self.async_run(boiler) for boiler in boilers]
        return await asyncio.gather(*tasks)

    def _get_efficiency(
        self, payload: Dict[str, Any], boiler_type: str, fuel_type: str
    ) -> float:
        """Determine boiler efficiency from input or defaults.

        Args:
            payload: Input payload
            boiler_type: Type of boiler (e.g., condensing, standard)
            fuel_type: Type of fuel used

        Returns:
            float: Efficiency value (0-1 scale)
        """
        if "efficiency" in payload:
            return payload["efficiency"]

        # Use defaults based on boiler type and age
        age = payload.get("age", "medium")

        if fuel_type in self.BOILER_EFFICIENCIES:
            fuel_efficiencies = self.BOILER_EFFICIENCIES[fuel_type]
            if boiler_type in fuel_efficiencies:
                return fuel_efficiencies[boiler_type].get(age, 0.75)

        # Default efficiency if not found in lookup
        return 0.75

    def _calculate_fuel_from_thermal(
        self, thermal_output: Dict[str, Any], efficiency: float, fuel_type: str
    ) -> Dict[str, Any]:
        """Calculate fuel consumption from thermal output using unit converter.

        Args:
            thermal_output: Thermal output with value and unit
            efficiency: Boiler efficiency (0-1 scale)
            fuel_type: Type of fuel

        Returns:
            Dict containing fuel consumption value and unit
        """
        # Convert thermal output to MMBtu using unit converter
        thermal_mmbtu = self.unit_converter.convert_energy(
            thermal_output["value"], thermal_output["unit"], "MMBtu"
        )

        # Calculate fuel input needed
        fuel_mmbtu = thermal_mmbtu / efficiency if efficiency > 0 else thermal_mmbtu

        # Convert to appropriate fuel unit
        if fuel_type == "natural_gas":
            return {"value": fuel_mmbtu * 10, "unit": "therms"}  # MMBtu to therms
        elif fuel_type in ["oil", "fuel_oil", "heating_oil"]:
            return {
                "value": fuel_mmbtu * 7.15,
                "unit": "gallons",
            }  # MMBtu to gallons (heating oil)
        elif fuel_type == "diesel":
            return {
                "value": fuel_mmbtu * 7.25,
                "unit": "gallons",
            }  # MMBtu to gallons (diesel)
        elif fuel_type == "propane":
            return {
                "value": fuel_mmbtu * 10.92,
                "unit": "gallons",
            }  # MMBtu to gallons (propane)
        elif fuel_type == "electricity":
            return {"value": fuel_mmbtu / 0.003412, "unit": "kWh"}  # MMBtu to kWh
        else:
            return {"value": fuel_mmbtu, "unit": "MMBtu"}

    def _calculate_thermal_from_fuel(
        self, fuel_consumption: Dict[str, Any], efficiency: float, fuel_type: str
    ) -> Dict[str, Any]:
        """Calculate thermal output from fuel consumption using unit converter.

        Args:
            fuel_consumption: Fuel consumption with value and unit
            efficiency: Boiler efficiency (0-1 scale)
            fuel_type: Type of fuel

        Returns:
            Dict containing thermal output value and unit
        """
        # Convert fuel to MMBtu using unit converter
        fuel_mmbtu = self.unit_converter.convert_fuel_to_energy(
            fuel_consumption["value"], fuel_consumption["unit"], fuel_type
        )

        # Calculate thermal output
        thermal_mmbtu = fuel_mmbtu * efficiency

        # Return in MMBtu
        return {"value": thermal_mmbtu, "unit": "MMBtu"}

    def _get_performance_rating(
        self, efficiency: float, boiler_type: str, fuel_type: str
    ) -> str:
        """Determine performance rating based on efficiency.

        Args:
            efficiency: Boiler efficiency (0-1 scale)
            boiler_type: Type of boiler
            fuel_type: Type of fuel

        Returns:
            str: Performance rating (Excellent/Good/Average/Poor)
        """
        if fuel_type == "natural_gas":
            if efficiency >= 0.90:
                return "Excellent"
            elif efficiency >= 0.80:
                return "Good"
            elif efficiency >= 0.70:
                return "Average"
            else:
                return "Poor"
        elif fuel_type == "oil":
            if efficiency >= 0.85:
                return "Excellent"
            elif efficiency >= 0.75:
                return "Good"
            elif efficiency >= 0.65:
                return "Average"
            else:
                return "Poor"
        elif fuel_type == "electric":
            if boiler_type == "heat_pump" and efficiency >= 3.0:
                return "Excellent"
            elif boiler_type == "heat_pump" and efficiency >= 2.5:
                return "Good"
            elif efficiency >= 0.95:
                return "Average"
            else:
                return "Poor"
        else:
            # Generic rating
            if efficiency >= 0.85:
                return "Excellent"
            elif efficiency >= 0.75:
                return "Good"
            elif efficiency >= 0.65:
                return "Average"
            else:
                return "Poor"

    def _generate_recommendations(
        self, boiler_type: str, fuel_type: str, efficiency: float, age: str
    ) -> List[Dict[str, str]]:
        """Generate optimization recommendations for the boiler.

        Args:
            boiler_type: Type of boiler
            fuel_type: Type of fuel
            efficiency: Current efficiency
            age: Age category of boiler

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Efficiency-based recommendations
        if efficiency < 0.70:
            recommendations.append(
                {
                    "priority": "high",
                    "action": "Replace boiler with high-efficiency condensing model",
                    "impact": "30-40% emissions reduction",
                    "payback": "3-5 years",
                }
            )
        elif efficiency < 0.80:
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "Upgrade to modern efficient boiler",
                    "impact": "15-25% emissions reduction",
                    "payback": "4-6 years",
                }
            )

        # Maintenance recommendations
        if age in ["old", "medium"]:
            recommendations.append(
                {
                    "priority": "high",
                    "action": "Perform comprehensive boiler tune-up and cleaning",
                    "impact": "5-10% efficiency improvement",
                    "payback": "< 1 year",
                }
            )

        # Fuel switching recommendations
        if fuel_type == "oil":
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "Consider switching to natural gas if available",
                    "impact": "20-30% emissions reduction",
                    "payback": "2-4 years",
                }
            )
        elif fuel_type == "coal":
            recommendations.append(
                {
                    "priority": "high",
                    "action": "Switch to cleaner fuel source (gas/biomass)",
                    "impact": "40-50% emissions reduction",
                    "payback": "3-5 years",
                }
            )

        # Control system recommendations
        recommendations.append(
            {
                "priority": "medium",
                "action": "Install smart boiler controls and weather compensation",
                "impact": "10-15% fuel savings",
                "payback": "2-3 years",
            }
        )

        # Heat recovery recommendations
        if boiler_type != "condensing" and fuel_type in ["natural_gas", "oil"]:
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "Install flue gas heat recovery system",
                    "impact": "5-8% efficiency improvement",
                    "payback": "3-4 years",
                }
            )

        # Insulation recommendations
        recommendations.append(
            {
                "priority": "low",
                "action": "Improve boiler and pipe insulation",
                "impact": "2-5% heat loss reduction",
                "payback": "1-2 years",
            }
        )

        return recommendations[:5]  # Return top 5 recommendations

    def _track_historical_data(self, data: Dict[str, Any]):
        """Track historical performance data for trend analysis.

        Args:
            data: Performance data to track
        """
        self._historical_data.append(data)

        # Keep only last 1000 records to avoid memory issues
        if len(self._historical_data) > 1000:
            self._historical_data = self._historical_data[-1000:]

    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate for performance monitoring.

        Returns:
            float: Cache hit rate (0-1)
        """
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    def export_results(self, results: List[BoilerOutput], format: str = "json") -> str:
        """Export results to various formats.

        Args:
            results: List of boiler outputs
            format: Export format (json, csv, excel)

        Returns:
            str: File path of exported data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            output_path = f"boiler_results_{timestamp}.json"
            with open(output_path, "w") as f:
                json.dump(
                    [r.dict() if hasattr(r, "dict") else r for r in results],
                    f,
                    indent=2,
                )

        elif format == "csv":
            import csv

            output_path = f"boiler_results_{timestamp}.csv"

            if results:
                keys = (
                    results[0].keys()
                    if isinstance(results[0], dict)
                    else results[0].__dict__.keys()
                )
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for result in results:
                        writer.writerow(
                            result if isinstance(result, dict) else result.__dict__
                        )

        elif format == "excel":
            try:
                import pandas as pd

                output_path = f"boiler_results_{timestamp}.xlsx"
                df = pd.DataFrame(results)
                df.to_excel(output_path, index=False)
            except ImportError:
                self.logger.error(
                    "pandas is required for Excel export. "
                    "Install it with: pip install greenlang[analytics]. "
                    "Falling back to CSV export."
                )
                return self.export_results(results, "csv")

        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Results exported to {output_path}")
        return output_path

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring.

        Returns:
            Dict containing performance metrics
        """
        avg_execution_time = (
            sum(self._execution_times) / len(self._execution_times)
            if self._execution_times
            else 0
        )

        return {
            "average_execution_time_ms": avg_execution_time * 1000,
            "total_calculations": len(self._execution_times),
            "cache_hit_rate": self._get_cache_hit_rate(),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "historical_records": len(self._historical_data),
        }

    def clear_cache(self):
        """Clear all caches and reset performance metrics."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._get_cached_emission_factor.cache_clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.info("Cache cleared")
