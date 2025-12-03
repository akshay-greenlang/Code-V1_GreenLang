"""
Domain-specific base agent classes.

Provides specialized base classes for common agent types:
- CalculatorAgentBase: Zero-hallucination calculations
- ValidatorAgentBase: Data validation
- RegulatoryAgentBase: Compliance checking
- ReportingAgentBase: Report generation
- IntegrationAgentBase: External system integration
- OrchestratorAgentBase: Multi-agent coordination
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum

from greenlang_sdk.core.agent_base import SDKAgentBase


class CalculationUnit(str, Enum):
    """Allowed units for climate calculations (zero-hallucination enforcement)."""
    KWH = "kWh"
    MWH = "MWh"
    GWH = "GWh"
    BTU = "BTU"
    MMBTU = "MMBTU"
    JOULE = "J"
    MEGAJOULE = "MJ"
    GIGAJOULE = "GJ"
    TCO2E = "tCO2e"
    KGCO2E = "kgCO2e"
    MTCO2E = "MtCO2e"
    CELSIUS = "°C"
    KELVIN = "K"
    PERCENT = "%"


class CalculatorAgentBase(SDKAgentBase):
    """
    Base class for calculation agents with zero-hallucination guarantees.

    Enforces:
    - All calculations use deterministic tools (NO LLM in calculation path)
    - Unit validation (only allowed climate units)
    - Complete provenance tracking
    - Automatic citation of emission factors
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_units = set(CalculationUnit)

    @abstractmethod
    def get_calculation_parameters(self) -> Dict[str, Any]:
        """
        Define required calculation parameters.

        Returns:
            Dictionary of parameter definitions
        """
        pass

    def validate_unit(self, unit: str) -> bool:
        """Validate that unit is in allowed climate units."""
        return unit in [u.value for u in self.allowed_units]

    async def validate_calculation_result(self, result: Any) -> Any:
        """
        Validate calculation result.

        Override to add domain-specific validation.
        """
        return result

    async def validate_output(self, output: Any, context: dict) -> Any:
        """Automatically validate all calculation results."""
        output = await super().validate_output(output, context)
        output = await self.validate_calculation_result(output)
        return output


class ValidatorAgentBase(SDKAgentBase):
    """
    Base class for validation agents.

    Enforces:
    - Schema validation
    - Constraint checking
    - Data quality validation
    """

    @abstractmethod
    def get_validation_schema(self) -> Dict[str, Any]:
        """
        Define validation schema.

        Returns:
            JSON Schema or Pydantic model definition
        """
        pass

    @abstractmethod
    async def validate_constraints(self, data: Any, context: dict) -> List[str]:
        """
        Validate business constraints.

        Returns:
            List of validation errors (empty if valid)
        """
        pass


class RegulatoryAgentBase(SDKAgentBase):
    """
    Base class for regulatory compliance agents.

    Enforces:
    - Regulation-specific validation
    - Compliance checking
    - Regulatory framework mapping (CBAM, CSRD, EUDR, etc.)
    """

    class Regulation(str, Enum):
        """Supported regulatory frameworks."""
        CBAM = "CBAM"
        CSRD = "CSRD"
        EUDR = "EUDR"
        SB253 = "SB253"
        GHG_PROTOCOL = "GHG_PROTOCOL"
        EU_TAXONOMY = "EU_TAXONOMY"
        VCCI = "VCCI"

    def __init__(self, regulation: Regulation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regulation = regulation

    @abstractmethod
    def get_regulatory_requirements(self) -> List[str]:
        """
        Get list of regulatory requirements.

        Returns:
            List of requirement identifiers
        """
        pass

    @abstractmethod
    async def check_compliance(self, data: Any, context: dict) -> Dict[str, bool]:
        """
        Check compliance against regulatory requirements.

        Returns:
            Dictionary mapping requirement IDs to compliance status
        """
        pass


class ReportingAgentBase(SDKAgentBase):
    """
    Base class for report generation agents.

    Supports:
    - Multi-format output (PDF, Excel, JSON, HTML)
    - Template-based generation
    - Regulatory report formats
    """

    class ReportFormat(str, Enum):
        """Supported report formats."""
        PDF = "pdf"
        EXCEL = "xlsx"
        JSON = "json"
        HTML = "html"
        CSV = "csv"

    @abstractmethod
    def get_report_template(self) -> str:
        """
        Get report template path or content.

        Returns:
            Template identifier or content
        """
        pass

    @abstractmethod
    async def generate_report(
        self,
        data: Any,
        format: ReportFormat,
        context: dict
    ) -> bytes:
        """
        Generate report in specified format.

        Returns:
            Report content as bytes
        """
        pass


class IntegrationAgentBase(SDKAgentBase):
    """
    Base class for external system integration agents.

    Supports:
    - ERP integration (SAP, Oracle, Workday)
    - SCADA integration
    - CMMS integration
    - API authentication and retry logic
    """

    @abstractmethod
    def get_connection_config(self) -> Dict[str, Any]:
        """
        Get connection configuration.

        Returns:
            Connection parameters (host, auth, etc.)
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test connection to external system.

        Returns:
            True if connection successful
        """
        pass


class OrchestratorAgentBase(SDKAgentBase):
    """
    Base class for multi-agent orchestration.

    Supports:
    - Sequential agent execution
    - Parallel agent execution
    - Conditional routing
    - Result aggregation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_agents: Dict[str, SDKAgentBase] = {}

    def register_agent(self, name: str, agent: SDKAgentBase):
        """Register a sub-agent."""
        self.sub_agents[name] = agent

    @abstractmethod
    async def orchestrate(self, input_data: Any, context: dict) -> Any:
        """
        Orchestrate sub-agent execution.

        Returns:
            Aggregated results
        """
        pass

    async def execute_sequential(
        self,
        agents: List[str],
        input_data: Any,
        context: dict
    ) -> List[Any]:
        """Execute agents sequentially."""
        results = []
        current_input = input_data

        for agent_name in agents:
            agent = self.sub_agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent not found: {agent_name}")

            result = await agent.run(current_input, context)
            results.append(result)
            current_input = result.output

        return results

    async def execute_parallel(
        self,
        agents: List[str],
        input_data: Any,
        context: dict
    ) -> List[Any]:
        """Execute agents in parallel."""
        import asyncio

        tasks = []
        for agent_name in agents:
            agent = self.sub_agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent not found: {agent_name}")

            tasks.append(agent.run(input_data, context))

        return await asyncio.gather(*tasks)
