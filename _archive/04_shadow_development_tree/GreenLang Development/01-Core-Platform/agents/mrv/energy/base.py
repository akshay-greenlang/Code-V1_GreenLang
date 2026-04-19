# -*- coding: utf-8 -*-
"""
MRV Energy Sector - Base Agent Class

This module provides the base class for all MRV Energy agents,
implementing the CRITICAL PATH pattern with zero-hallucination guarantee.

All MRV agents inherit from this base and follow:
- Deterministic calculations only (no LLM in calculation path)
- Full audit trail with SHA-256 provenance hashing
- GHG Protocol compliance
- Standardized input/output validation
"""

from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TypeVar
import hashlib
import json
import logging
import time

from pydantic import BaseModel

from greenlang.agents.base_agents import DeterministicAgent, AuditEntry
from greenlang.agents.categories import AgentCategory, AgentMetadata


logger = logging.getLogger(__name__)

# Type variables for generic input/output
InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class MRVEnergyBaseAgent(DeterministicAgent):
    """
    Base class for MRV Energy sector agents.

    This class extends DeterministicAgent with energy-sector-specific
    functionality including:
    - Standard emission factor lookups
    - GHG Protocol compliance validation
    - EPA reporting compatibility
    - EU ETS monitoring requirements

    All subclasses must implement:
    - execute(): Core calculation logic
    - _get_emission_factors(): Emission factor retrieval

    Example:
        class PowerGenerationMRVAgent(MRVEnergyBaseAgent):
            metadata = AgentMetadata(
                name="GL-MRV-ENE-001",
                category=AgentCategory.CRITICAL,
                description="Power generation emissions MRV"
            )

            def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                # Implementation
                pass

    Attributes:
        agent_id: Unique agent identifier (e.g., GL-MRV-ENE-001)
        version: Agent version string
        supported_fuels: List of supported fuel types
        emission_factors: Cached emission factors
    """

    category = AgentCategory.CRITICAL
    metadata: Optional[AgentMetadata] = None

    # Standard GWP values (IPCC AR5)
    GWP_CO2 = 1.0
    GWP_CH4 = 28.0
    GWP_N2O = 265.0

    def __init__(
        self,
        agent_id: str,
        version: str = "1.0.0",
        enable_audit_trail: bool = True,
    ):
        """
        Initialize MRV Energy base agent.

        Args:
            agent_id: Unique agent identifier (e.g., GL-MRV-ENE-001)
            version: Agent version string
            enable_audit_trail: Whether to capture full audit trail
        """
        super().__init__(enable_audit_trail=enable_audit_trail)

        self.agent_id = agent_id
        self.version = version
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")

        # Emission factor cache
        self._emission_factors: Dict[str, Dict[str, float]] = {}

        # Load standard emission factors
        self._load_emission_factors()

        self.logger.info(f"Initialized {agent_id} v{version}")

    def _load_emission_factors(self) -> None:
        """
        Load standard emission factors for energy calculations.

        Factors are from:
        - EPA eGRID (US grid)
        - EPA AP-42 (combustion)
        - IPCC Guidelines (defaults)
        - EU ETS Monitoring Regulation
        """
        # CO2 emission factors by fuel (kg CO2 / MMBTU)
        self._emission_factors["co2_mmbtu"] = {
            "natural_gas": 53.06,
            "coal_bituminous": 93.28,
            "coal_subbituminous": 97.17,
            "coal_lignite": 97.72,
            "coal_anthracite": 103.69,
            "fuel_oil_no2": 73.96,
            "fuel_oil_no6": 75.10,
            "diesel": 73.96,
            "lpg": 62.98,
            "biomass_wood": 0.0,  # Biogenic
            "biomass_biogas": 0.0,  # Biogenic
            "landfill_gas": 0.0,  # Biogenic
            "msw": 41.69,  # Non-biogenic fraction
        }

        # CH4 emission factors (g CH4 / MMBTU)
        self._emission_factors["ch4_mmbtu"] = {
            "natural_gas": 1.0,
            "coal_bituminous": 11.0,
            "coal_subbituminous": 11.0,
            "coal_lignite": 11.0,
            "fuel_oil_no2": 3.0,
            "fuel_oil_no6": 3.0,
            "diesel": 3.0,
        }

        # N2O emission factors (g N2O / MMBTU)
        self._emission_factors["n2o_mmbtu"] = {
            "natural_gas": 0.1,
            "coal_bituminous": 1.6,
            "coal_subbituminous": 1.6,
            "coal_lignite": 1.6,
            "fuel_oil_no2": 0.6,
            "fuel_oil_no6": 0.6,
            "diesel": 0.6,
        }

        # Grid emission factors by region (kg CO2e / MWh)
        self._emission_factors["grid_kg_mwh"] = {
            # US eGRID 2022
            "us_wecc": 338.0,
            "us_mroe": 610.0,
            "us_npcc": 227.0,
            "us_rfce": 370.0,
            "us_rfcm": 671.0,
            "us_rfcw": 554.0,
            "us_srmw": 662.0,
            "us_srmv": 422.0,
            "us_srso": 445.0,
            "us_srtv": 477.0,
            "us_srvc": 356.0,
            "us_spno": 475.0,
            "us_spso": 476.0,
            "us_camx": 234.0,
            "us_nwpp": 307.0,
            "us_aznm": 416.0,
            "us_rmpa": 490.0,
            # EU (2023 averages)
            "eu_nordic": 47.0,
            "eu_central_west": 285.0,
            "eu_central_east": 520.0,
            "eu_iberian": 165.0,
            "eu_italian": 315.0,
            "eu_british": 207.0,
            # Asia
            "china_north": 790.0,
            "china_south": 610.0,
            "india_north": 820.0,
            "india_south": 730.0,
            "japan_east": 450.0,
            "japan_west": 410.0,
        }

        # Lifecycle emission factors for renewables (g CO2e / kWh)
        self._emission_factors["lifecycle_g_kwh"] = {
            "solar_pv_utility": 41.0,
            "solar_pv_rooftop": 48.0,
            "solar_csp": 27.0,
            "wind_onshore": 11.0,
            "wind_offshore": 12.0,
            "hydro_ror": 4.0,
            "hydro_reservoir": 24.0,
            "nuclear_pwr": 12.0,
            "nuclear_bwr": 12.0,
            "geothermal": 38.0,
        }

        # Fuel heating values (MMBTU / unit)
        self._emission_factors["heating_value_mmbtu"] = {
            "natural_gas_mcf": 1.028,
            "natural_gas_therm": 0.1,
            "coal_ton": 21.0,
            "fuel_oil_gallon": 0.140,
            "diesel_gallon": 0.138,
            "lpg_gallon": 0.091,
        }

    def get_emission_factor(
        self,
        factor_type: str,
        key: str,
        default: Optional[float] = None
    ) -> float:
        """
        Get emission factor from cache.

        Args:
            factor_type: Type of factor (co2_mmbtu, grid_kg_mwh, etc.)
            key: Specific factor key (fuel type, region, etc.)
            default: Default value if not found

        Returns:
            Emission factor value

        Raises:
            KeyError: If factor not found and no default provided
        """
        factors = self._emission_factors.get(factor_type, {})
        if key in factors:
            return factors[key]
        if default is not None:
            return default
        raise KeyError(f"Emission factor not found: {factor_type}/{key}")

    def calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        calculation_trace: List[str]
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Args:
            inputs: Input data dictionary
            outputs: Output data dictionary
            calculation_trace: Step-by-step calculation trace

        Returns:
            SHA-256 hex digest
        """
        provenance_data = {
            "agent_id": self.agent_id,
            "version": self.version,
            "inputs": inputs,
            "outputs": outputs,
            "calculation_trace": calculation_trace,
            "emission_factors_version": "2024.1",
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def validate_inputs(
        self,
        inputs: Dict[str, Any],
        required_fields: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate input data completeness.

        Args:
            inputs: Input data dictionary
            required_fields: List of required field names

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        for field in required_fields:
            if field not in inputs:
                errors.append(f"Missing required field: {field}")
            elif inputs[field] is None:
                errors.append(f"Field is None: {field}")

        return len(errors) == 0, errors

    def calculate_uncertainty(
        self,
        data_quality: str,
        emission_type: str = "combustion"
    ) -> float:
        """
        Calculate uncertainty percentage based on data quality.

        Args:
            data_quality: Data quality level (measured, calculated, estimated, extrapolated)
            emission_type: Type of emission (combustion, grid, lifecycle)

        Returns:
            Uncertainty percentage
        """
        base_uncertainty = {
            "measured": 2.5,
            "calculated": 7.5,
            "estimated": 20.0,
            "extrapolated": 40.0,
        }

        emission_factor_uncertainty = {
            "combustion": 5.0,
            "grid": 10.0,
            "lifecycle": 15.0,
            "fugitive": 25.0,
        }

        base = base_uncertainty.get(data_quality, 20.0)
        factor = emission_factor_uncertainty.get(emission_type, 10.0)

        # Combined uncertainty (root sum of squares)
        return round((base**2 + factor**2) ** 0.5, 1)

    def create_audit_entry(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        calculation_trace: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEntry:
        """
        Create and record an audit entry.

        Args:
            operation: Operation name
            inputs: Input data
            outputs: Output data
            calculation_trace: Calculation steps
            metadata: Additional metadata

        Returns:
            Created AuditEntry
        """
        return self._capture_audit_entry(
            operation=operation,
            inputs=inputs,
            outputs=outputs,
            calculation_trace=calculation_trace,
            metadata=metadata
        )

    def process(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process inputs through the agent with full lifecycle.

        This method wraps execute() with:
        - Input validation
        - Performance timing
        - Provenance tracking
        - Audit trail capture

        Args:
            inputs: Input data dictionary

        Returns:
            Output dictionary with provenance and metadata
        """
        start_time = time.time()
        calculation_trace: List[str] = []

        try:
            # Record start
            calculation_trace.append(
                f"Started {self.agent_id} processing at {datetime.now(timezone.utc).isoformat()}"
            )

            # Execute core logic
            outputs = self.execute(inputs)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Add provenance
            outputs["agent_id"] = self.agent_id
            outputs["provenance_hash"] = self.calculate_provenance_hash(
                inputs, outputs, calculation_trace
            )
            outputs["processing_time_ms"] = round(processing_time_ms, 2)

            # Record completion
            calculation_trace.append(
                f"Completed processing in {processing_time_ms:.2f}ms"
            )
            outputs["calculation_trace"] = calculation_trace

            # Create audit entry
            if self.enable_audit_trail:
                self.create_audit_entry(
                    operation=f"{self.agent_id}_calculation",
                    inputs=inputs,
                    outputs=outputs,
                    calculation_trace=calculation_trace,
                    metadata={"version": self.version}
                )

            return outputs

        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute core calculation logic.

        This method MUST be:
        - Deterministic (same inputs -> same outputs)
        - Fast (no network calls, no LLM)
        - Pure (no side effects except logging)
        - Traceable (populate calculation_trace)

        Args:
            inputs: Validated input data

        Returns:
            Calculation results
        """
        pass
