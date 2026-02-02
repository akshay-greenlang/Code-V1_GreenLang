# Base Agent Interface - Complete Specification

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** Specification

## Executive Summary

This document specifies the `SDKAgentBase` interface, which extends `AgentSpecV2Base` with domain-specific features for GreenLang agents. The interface establishes standard lifecycle methods, metadata fields, error handling patterns, and integration patterns for building production-grade climate and regulatory agents.

---

## Table of Contents

1. [Interface Overview](#interface-overview)
2. [Class Hierarchy](#class-hierarchy)
3. [Lifecycle Methods](#lifecycle-methods)
4. [Metadata Fields](#metadata-fields)
5. [Error Handling Patterns](#error-handling-patterns)
6. [Tool Integration](#tool-integration)
7. [Provenance Tracking](#provenance-tracking)
8. [Testing Support](#testing-support)
9. [Implementation Examples](#implementation-examples)

---

## Interface Overview

### Purpose

`SDKAgentBase` provides:
- Enhanced AgentSpec v2 lifecycle with domain-specific features
- Standard metadata for climate/regulatory domains
- Tool execution patterns for zero-hallucination guarantee
- Comprehensive error handling and logging
- Provenance tracking with SHA-256 hashing
- Testing utilities and fixtures

### Design Goals

1. **Enhance, Don't Replace**: Build on AgentSpec v2, not replace it
2. **Domain-Specific**: Climate, energy, regulatory patterns
3. **Zero-Hallucination**: Tool-first architecture enforcement
4. **Type-Safe**: Full Pydantic model integration
5. **Production-Ready**: Comprehensive error handling, logging, monitoring

### Key Features

```python
# Standard lifecycle with domain features
agent = SDKAgentBase()
agent.initialize()           # Load pack.yaml, setup tools
agent.validate_input(input)  # Schema + domain validation
agent.execute(input)         # Tool-based calculation
agent.validate_output(out)   # Output validation
agent.finalize(result)       # Add provenance, citations

# Domain-specific metadata
agent.domain          # "emissions", "energy", "regulatory"
agent.regulations     # ["EPA CEMS", "EU ETS", "GRI 305"]
agent.tool_registry   # Available tools
agent.ef_database     # Emission factor database

# Zero-hallucination enforcement
agent.use_tool("calculator", params)  # Deterministic tools only
agent.validate_calculation(result)    # Verify tool output
agent.track_provenance(input, output) # SHA-256 hashing
```

---

## Class Hierarchy

### Inheritance Model

```
AgentSpecV2Base[InT, OutT]         # Core framework (existing)
    ↓
SDKAgentBase[InT, OutT]            # SDK enhancements (new)
    ↓
CalculatorAgentBase[InT, OutT]     # Domain-specific base (new)
    ↓
CarbonCalculatorAgent              # Concrete implementation
```

### Class Definition

```python
"""
SDK Agent Base Class
====================

Enhanced AgentSpec v2 base with domain-specific features.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Any, Generic, TypeVar
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from greenlang.agents.agentspec_v2_base import (
    AgentSpecV2Base,
    AgentExecutionContext,
    AgentLifecycleState,
)
from greenlang.types import AgentResult
from greenlang_sdk.tools import ToolRegistry, ToolResult
from greenlang_sdk.provenance import ProvenanceTracker
from greenlang_sdk.validation import DomainValidator
from greenlang_sdk.models import DomainMetadata, ToolExecution

logger = logging.getLogger(__name__)

# Type variables
InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


class SDKAgentBase(AgentSpecV2Base[InT, OutT]):
    """
    Enhanced AgentSpec v2 base with SDK features.

    This class extends AgentSpecV2Base with domain-specific features:
    - Tool registry and execution
    - Domain metadata (domain, regulations, frameworks)
    - Enhanced provenance tracking
    - Domain-specific validation
    - Testing utilities

    Example:
        >>> class MyAgent(SDKAgentBase[MyInput, MyOutput]):
        ...     def execute_impl(self, input: MyInput, context) -> MyOutput:
        ...         result = self.use_tool("calculator", {"value": input.amount})
        ...         return MyOutput(result=result.data["value"])
        ...
        >>> agent = MyAgent(pack_path=Path("packs/my_agent"))
        >>> result = agent.run(MyInput(amount=1000))
    """

    def __init__(
        self,
        pack_path: Optional[Path] = None,
        agent_id: Optional[str] = None,
        domain: Optional[str] = None,
        regulations: Optional[List[str]] = None,
        enable_tool_validation: bool = True,
        enable_provenance_tracking: bool = True,
        **kwargs
    ):
        """
        Initialize SDK Agent Base.

        Args:
            pack_path: Path to pack directory containing pack.yaml
            agent_id: Agent identifier (auto-detected from pack.yaml)
            domain: Domain category (emissions, energy, regulatory, etc.)
            regulations: List of applicable regulations (EPA, EU ETS, GRI, etc.)
            enable_tool_validation: Validate all tool executions
            enable_provenance_tracking: Track provenance with SHA-256
            **kwargs: Additional arguments for AgentSpecV2Base
        """
        # Call parent constructor
        super().__init__(
            pack_path=pack_path,
            agent_id=agent_id,
            **kwargs
        )

        # SDK-specific attributes
        self.domain = domain
        self.regulations = regulations or []
        self.enable_tool_validation = enable_tool_validation
        self.enable_provenance_tracking = enable_provenance_tracking

        # Tool management
        self.tool_registry = ToolRegistry()
        self.tool_executions: List[ToolExecution] = []

        # Provenance tracking
        self.provenance_tracker = ProvenanceTracker()

        # Domain validation
        self.domain_validator = DomainValidator(domain=self.domain)

        # Load domain metadata from pack.yaml
        self._load_domain_metadata()

    # =========================================================================
    # Enhanced Lifecycle Methods
    # =========================================================================

    def initialize_impl(self) -> None:
        """
        SDK-specific initialization.

        Override this method to add custom initialization logic.
        This is called after pack.yaml is loaded.
        """
        # Register default tools
        self._register_default_tools()

        # Initialize domain validator
        self._initialize_domain_validator()

        # Load emission factor database (if applicable)
        if self.domain in ["emissions", "energy", "carbon"]:
            self._load_emission_factors()

        logger.info(
            f"{self.agent_id} SDK initialization complete "
            f"(domain={self.domain}, regulations={len(self.regulations)})"
        )

    def validate_input_impl(
        self,
        input_data: InT,
        context: AgentExecutionContext
    ) -> InT:
        """
        Enhanced input validation with domain-specific checks.

        Args:
            input_data: Input data to validate
            context: Execution context

        Returns:
            Validated input data

        Raises:
            ValidationError: If validation fails
        """
        # Parent validation (schema, constraints)
        validated_input = super().validate_input_impl(input_data, context)

        # Domain-specific validation
        if self.domain:
            validation_result = self.domain_validator.validate(
                data=validated_input,
                validation_type="input"
            )

            if not validation_result.is_valid:
                raise ValueError(
                    f"Domain validation failed: {validation_result.errors}"
                )

        return validated_input

    def validate_output_impl(
        self,
        output: OutT,
        context: AgentExecutionContext
    ) -> OutT:
        """
        Enhanced output validation with domain-specific checks.

        Args:
            output: Output data to validate
            context: Execution context

        Returns:
            Validated output data

        Raises:
            ValidationError: If validation fails
        """
        # Parent validation
        validated_output = super().validate_output_impl(output, context)

        # Domain-specific validation
        if self.domain:
            validation_result = self.domain_validator.validate(
                data=validated_output,
                validation_type="output"
            )

            if not validation_result.is_valid:
                raise ValueError(
                    f"Output validation failed: {validation_result.errors}"
                )

        return validated_output

    def finalize_impl(
        self,
        result: AgentResult[OutT],
        context: AgentExecutionContext
    ) -> AgentResult[OutT]:
        """
        Enhanced finalization with provenance and tool execution logs.

        Args:
            result: Agent result to finalize
            context: Execution context

        Returns:
            Finalized result with SDK metadata
        """
        # Parent finalization
        result = super().finalize_impl(result, context)

        # Add tool execution logs
        if self.tool_executions:
            result.metadata["tool_executions"] = [
                {
                    "tool_id": te.tool_id,
                    "execution_time_ms": te.execution_time_ms,
                    "input_hash": te.input_hash,
                    "output_hash": te.output_hash,
                }
                for te in self.tool_executions
            ]

        # Add domain metadata
        result.metadata["domain"] = self.domain
        result.metadata["regulations"] = self.regulations

        # Add provenance summary
        if self.enable_provenance_tracking:
            result.metadata["provenance_hash"] = self.provenance_tracker.get_chain_hash()

        return result

    # =========================================================================
    # Tool Execution Methods
    # =========================================================================

    def use_tool(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        track_provenance: bool = True
    ) -> ToolResult:
        """
        Execute a tool with validation and provenance tracking.

        This is the primary method for executing deterministic tools.
        All calculations MUST go through this method to enforce
        zero-hallucination guarantee.

        Args:
            tool_id: Tool identifier (e.g., "calculator", "emission_factor_lookup")
            parameters: Tool input parameters
            track_provenance: Whether to track provenance for this execution

        Returns:
            ToolResult with data and provenance

        Raises:
            ToolNotFoundError: If tool not registered
            ToolExecutionError: If tool execution fails
            ValidationError: If parameters invalid

        Example:
            >>> result = self.use_tool(
            ...     "emissions_calculator",
            ...     {
            ...         "activity_data": 1000,
            ...         "emission_factor": 53.06,
            ...         "unit": "kg"
            ...     }
            ... )
            >>> emissions = result.data["co2e_emissions_kg"]
        """
        start_time = datetime.now()

        try:
            # Get tool from registry
            tool = self.tool_registry.get_tool(tool_id)

            # Validate parameters
            if self.enable_tool_validation:
                validated_params = tool.validate_parameters(parameters)
            else:
                validated_params = parameters

            # Execute tool
            result = tool.execute(validated_params)

            # Track execution
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            tool_execution = ToolExecution(
                tool_id=tool_id,
                parameters=validated_params,
                result=result.data,
                execution_time_ms=execution_time_ms,
                input_hash=result.provenance.get("input_hash", ""),
                output_hash=result.provenance.get("output_hash", ""),
                timestamp=datetime.now(),
            )
            self.tool_executions.append(tool_execution)

            # Track provenance
            if track_provenance and self.enable_provenance_tracking:
                self.provenance_tracker.add_execution(tool_execution)

            logger.debug(
                f"Tool {tool_id} executed successfully "
                f"(took {execution_time_ms:.2f}ms)"
            )

            return result

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_id}", exc_info=True)
            raise ToolExecutionError(f"Tool {tool_id} failed: {str(e)}") from e

    def register_tool(
        self,
        tool_id: str,
        tool: Any,
        override: bool = False
    ) -> None:
        """
        Register a tool in the tool registry.

        Args:
            tool_id: Unique tool identifier
            tool: Tool instance
            override: Whether to override existing tool

        Raises:
            ToolAlreadyRegisteredError: If tool exists and override=False
        """
        self.tool_registry.register(tool_id, tool, override=override)
        logger.info(f"Tool registered: {tool_id}")

    def get_tool(self, tool_id: str) -> Any:
        """
        Get a tool from the registry.

        Args:
            tool_id: Tool identifier

        Returns:
            Tool instance

        Raises:
            ToolNotFoundError: If tool not found
        """
        return self.tool_registry.get_tool(tool_id)

    def list_tools(self) -> List[str]:
        """
        List all registered tools.

        Returns:
            List of tool identifiers
        """
        return self.tool_registry.list_tools()

    # =========================================================================
    # Provenance Tracking Methods
    # =========================================================================

    def track_provenance(
        self,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Track provenance for input/output pair.

        Args:
            input_data: Input data
            output_data: Output data
            metadata: Additional metadata

        Returns:
            SHA-256 hash of provenance record
        """
        return self.provenance_tracker.track(
            input_data=input_data,
            output_data=output_data,
            metadata=metadata
        )

    def get_provenance_chain(self) -> List[Dict]:
        """
        Get complete provenance chain for this execution.

        Returns:
            List of provenance records
        """
        return self.provenance_tracker.get_chain()

    def validate_provenance(self) -> bool:
        """
        Validate integrity of provenance chain.

        Returns:
            True if chain is valid, False otherwise
        """
        return self.provenance_tracker.validate_chain()

    # =========================================================================
    # Domain-Specific Methods
    # =========================================================================

    def get_emission_factor(
        self,
        material_id: str,
        region: Optional[str] = None,
        year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get emission factor from database with provenance.

        Args:
            material_id: Material/fuel identifier
            region: Geographic region (optional)
            year: Year for EF (optional, defaults to latest)

        Returns:
            Emission factor with metadata and provenance

        Example:
            >>> ef = self.get_emission_factor("natural_gas", region="US")
            >>> value = ef["value"]  # 53.06 kgCO2e/mmBTU
            >>> source = ef["source"]  # "EPA 2023"
        """
        result = self.use_tool(
            "emission_factor_lookup",
            {
                "material_id": material_id,
                "region": region,
                "year": year
            }
        )
        return result.data

    def convert_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert between climate units.

        Args:
            value: Value to convert
            from_unit: Source unit (e.g., "kWh", "BTU", "GJ")
            to_unit: Target unit

        Returns:
            Converted value

        Example:
            >>> kwh_value = self.convert_units(1000, "BTU", "kWh")
        """
        result = self.use_tool(
            "unit_converter",
            {
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit
            }
        )
        return result.data["converted_value"]

    def map_to_framework(
        self,
        data: Dict[str, Any],
        framework: str
    ) -> Dict[str, Any]:
        """
        Map data to regulatory framework format.

        Args:
            data: Data to map
            framework: Framework identifier (GRI, SASB, TCFD, CDP)

        Returns:
            Framework-compliant data structure

        Example:
            >>> gri_data = self.map_to_framework(emissions_data, "GRI_305")
        """
        result = self.use_tool(
            "framework_mapper",
            {
                "data": data,
                "framework": framework
            }
        )
        return result.data

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _load_domain_metadata(self) -> None:
        """Load domain metadata from pack.yaml."""
        if not self.spec:
            return

        # Extract domain from pack.yaml metadata
        if hasattr(self.spec, "metadata") and self.spec.metadata:
            self.domain = self.spec.metadata.get("domain", self.domain)
            self.regulations = self.spec.metadata.get("regulations", self.regulations)

        # Extract from tags
        if hasattr(self.spec, "tags") and self.spec.tags:
            # Look for domain tags
            domain_tags = [
                "emissions", "energy", "water", "waste",
                "carbon", "climate", "esg", "regulatory"
            ]
            for tag in self.spec.tags:
                if tag.lower() in domain_tags and not self.domain:
                    self.domain = tag.lower()
                    break

    def _register_default_tools(self) -> None:
        """Register default tools for all agents."""
        from greenlang_sdk.tools import (
            UnitConverter,
            EmissionFactorLookup,
            FrameworkMapper,
            DataValidator,
        )

        # Always available tools
        self.register_tool("unit_converter", UnitConverter())
        self.register_tool("emission_factor_lookup", EmissionFactorLookup())
        self.register_tool("framework_mapper", FrameworkMapper())
        self.register_tool("data_validator", DataValidator())

    def _initialize_domain_validator(self) -> None:
        """Initialize domain-specific validation rules."""
        if self.domain:
            self.domain_validator.load_rules(self.domain)

    def _load_emission_factors(self) -> None:
        """Load emission factor database for emissions/energy domains."""
        # Load EF database from configured source
        pass

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_metadata(self) -> DomainMetadata:
        """
        Get complete agent metadata.

        Returns:
            DomainMetadata with all agent information
        """
        return DomainMetadata(
            agent_id=self.agent_id,
            domain=self.domain,
            regulations=self.regulations,
            tools=self.list_tools(),
            execution_count=self._execution_count,
            total_execution_time_ms=self._total_execution_time_ms,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"agent_id={self.agent_id}, "
            f"domain={self.domain}, "
            f"tools={len(self.list_tools())}, "
            f"executions={self._execution_count})"
        )
```

---

## Lifecycle Methods

### Standard Lifecycle Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SDK Agent Lifecycle                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. initialize()                                                  │
│     ├── Load pack.yaml (AgentSpec v2)                           │
│     ├── initialize_impl() (SDK)                                  │
│     │   ├── Register default tools                               │
│     │   ├── Initialize domain validator                          │
│     │   └── Load emission factors (if applicable)                │
│     └── Set state = INITIALIZED                                  │
│                                                                   │
│  2. validate_input(input_data, context)                          │
│     ├── Schema validation (AgentSpec v2)                         │
│     ├── validate_input_impl() (SDK)                              │
│     │   └── Domain-specific validation                           │
│     └── Return validated input                                   │
│                                                                   │
│  3. execute(validated_input, context)                            │
│     ├── execute_impl() (Subclass)                                │
│     │   ├── use_tool() for calculations                          │
│     │   ├── Track provenance                                     │
│     │   └── Generate output                                      │
│     └── Track execution time                                     │
│                                                                   │
│  4. validate_output(output, context)                             │
│     ├── Schema validation (AgentSpec v2)                         │
│     ├── validate_output_impl() (SDK)                             │
│     │   └── Domain-specific validation                           │
│     └── Return validated output                                  │
│                                                                   │
│  5. finalize(result, context)                                    │
│     ├── Add citations (AgentSpec v2)                             │
│     ├── finalize_impl() (SDK)                                    │
│     │   ├── Add tool execution logs                              │
│     │   ├── Add domain metadata                                  │
│     │   └── Add provenance hash                                  │
│     └── Return finalized result                                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Method Signatures

```python
# Initialization
def initialize(self) -> None:
    """Load pack.yaml, setup tools, initialize validators."""

def initialize_impl(self) -> None:
    """SDK-specific initialization (override in subclass)."""

# Validation
def validate_input(self, input_data: InT, context: AgentExecutionContext) -> InT:
    """Validate input with schema + domain checks."""

def validate_input_impl(self, input_data: InT, context: AgentExecutionContext) -> InT:
    """Domain-specific input validation (override in subclass)."""

def validate_output(self, output: OutT, context: AgentExecutionContext) -> OutT:
    """Validate output with schema + domain checks."""

def validate_output_impl(self, output: OutT, context: AgentExecutionContext) -> OutT:
    """Domain-specific output validation (override in subclass)."""

# Execution
def execute(self, validated_input: InT, context: AgentExecutionContext) -> OutT:
    """Execute agent logic with timing and error handling."""

def execute_impl(self, validated_input: InT, context: AgentExecutionContext) -> OutT:
    """Core agent logic (MUST be implemented by subclass)."""

# Finalization
def finalize(self, result: AgentResult[OutT], context: AgentExecutionContext) -> AgentResult[OutT]:
    """Finalize result with metadata, provenance, citations."""

def finalize_impl(self, result: AgentResult[OutT], context: AgentExecutionContext) -> AgentResult[OutT]:
    """SDK-specific finalization (override in subclass)."""

# Main entry point
def run(self, payload: InT) -> AgentResult[OutT]:
    """Execute complete lifecycle: initialize → validate → execute → finalize."""
```

---

## Metadata Fields

### Domain Metadata

```python
class DomainMetadata(BaseModel):
    """Domain-specific metadata for SDK agents."""

    agent_id: str = Field(..., description="Agent identifier")
    domain: str = Field(..., description="Domain category")
    regulations: List[str] = Field(default_factory=list, description="Applicable regulations")
    frameworks: List[str] = Field(default_factory=list, description="Reporting frameworks")
    tools: List[str] = Field(default_factory=list, description="Registered tools")
    execution_count: int = Field(0, description="Number of executions")
    total_execution_time_ms: float = Field(0.0, description="Total execution time")
    provenance_hash: Optional[str] = Field(None, description="Provenance chain hash")
```

### Supported Domains

| Domain | Description | Common Regulations |
|--------|-------------|-------------------|
| `emissions` | GHG emissions calculations | EPA CEMS, EU ETS, GRI 305 |
| `energy` | Energy consumption/efficiency | ISO 50001, ASHRAE 90.1 |
| `water` | Water consumption/quality | EPA SDWA, ISO 46001 |
| `waste` | Waste generation/management | EPA RCRA, EU Waste Directive |
| `carbon` | Carbon accounting | GHG Protocol, ISO 14064 |
| `climate` | Climate risk assessment | TCFD, SASB |
| `esg` | ESG reporting | GRI, SASB, CDP |
| `regulatory` | Regulatory compliance | Framework-specific |

### Supported Regulations

```python
REGULATIONS = [
    # Emissions
    "EPA_CEMS",      # EPA Continuous Emissions Monitoring
    "EU_ETS",        # EU Emissions Trading System
    "GRI_305",       # GRI 305: Emissions
    "ISO_14064",     # ISO 14064: GHG Accounting

    # Energy
    "ISO_50001",     # ISO 50001: Energy Management
    "ASHRAE_90_1",   # ASHRAE 90.1: Energy Standard
    "EU_EED",        # EU Energy Efficiency Directive

    # Climate
    "TCFD",          # Task Force on Climate-related Financial Disclosures
    "SASB",          # Sustainability Accounting Standards Board
    "CDP",           # Carbon Disclosure Project

    # ESG
    "GRI_UNIVERSAL", # GRI Universal Standards
    "ISSB",          # International Sustainability Standards Board
    "CSRD",          # Corporate Sustainability Reporting Directive
]
```

---

## Error Handling Patterns

### Error Hierarchy

```python
GreenLangException                    # Base exception
├── ValidationError                   # Input/output validation
│   ├── SchemaValidationError        # Schema mismatch
│   ├── ConstraintValidationError    # Constraint violation
│   └── DomainValidationError        # Domain-specific validation
├── ExecutionError                    # Execution failures
│   ├── ToolExecutionError           # Tool execution failure
│   ├── CalculationError             # Calculation failure
│   └── IntegrationError             # External system failure
├── ProvenanceError                   # Provenance tracking
│   ├── ProvenanceValidationError    # Chain validation failure
│   └── ProvenanceTrackingError      # Tracking failure
└── ConfigurationError                # Configuration issues
    ├── ToolNotFoundError            # Tool not registered
    └── PackYAMLError                # pack.yaml issues
```

### Error Handling Pattern

```python
from greenlang_sdk.exceptions import (
    ValidationError,
    ToolExecutionError,
    IntegrationError,
    CriticalError,
)

def execute_impl(
    self,
    validated_input: InT,
    context: AgentExecutionContext
) -> OutT:
    """Execute with comprehensive error handling."""

    try:
        # Step 1: Validate input (fail fast)
        if not self._pre_execution_check(validated_input):
            raise ValidationError("Pre-execution check failed")

        # Step 2: Execute tool (log and handle)
        try:
            result = self.use_tool("calculator", {"value": validated_input.amount})
        except ToolExecutionError as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            # Return error result, don't crash
            return self._create_error_output(str(e))

        # Step 3: External integration (use fallback)
        try:
            external_data = self._fetch_from_erp(validated_input.id)
        except IntegrationError as e:
            logger.warning(f"Integration failed, using cached data: {e}")
            external_data = self._get_cached_data(validated_input.id)

        # Step 4: Combine results
        output = self._combine_results(result, external_data)

        # Step 5: Validate output
        if not self._validate_output_constraints(output):
            raise ValidationError("Output constraint violation")

        return output

    except ValidationError as e:
        # Fail fast - bad input/output
        logger.error(f"Validation error: {e}")
        raise

    except CriticalError as e:
        # Emergency shutdown - safety issue
        logger.critical(f"CRITICAL ERROR: {e}", exc_info=True)
        self._emergency_shutdown()
        raise

    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise ExecutionError(f"Execution failed: {str(e)}") from e
```

### Error Response Format

```python
class ErrorResult(BaseModel):
    """Standard error result format."""

    success: bool = Field(False, description="Always False for errors")
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict] = Field(None, description="Additional details")
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_id: str = Field(..., description="Agent identifier")
    execution_id: str = Field(..., description="Execution identifier")
```

---

## Tool Integration

### Tool Interface

```python
class Tool(ABC):
    """Base interface for all tools."""

    @property
    @abstractmethod
    def tool_id(self) -> str:
        """Unique tool identifier."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Dict:
        """JSON schema for input parameters."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Dict:
        """JSON schema for output data."""
        pass

    @abstractmethod
    def execute(self, parameters: Dict) -> ToolResult:
        """
        Execute tool with parameters.

        Args:
            parameters: Tool input parameters

        Returns:
            ToolResult with data and provenance
        """
        pass

    def validate_parameters(self, parameters: Dict) -> Dict:
        """Validate parameters against input schema."""
        # Implementation
        pass
```

### Tool Result

```python
class ToolResult(BaseModel):
    """Result of tool execution."""

    data: Dict[str, Any] = Field(..., description="Tool output data")
    provenance: Dict[str, Any] = Field(..., description="Provenance metadata")
    execution_time_ms: float = Field(..., description="Execution time")
    deterministic: bool = Field(True, description="Whether result is deterministic")
```

### Tool Registry

```python
class ToolRegistry:
    """Registry for managing tools."""

    def register(self, tool_id: str, tool: Tool, override: bool = False) -> None:
        """Register a tool."""
        pass

    def get_tool(self, tool_id: str) -> Tool:
        """Get a tool by ID."""
        pass

    def list_tools(self) -> List[str]:
        """List all registered tools."""
        pass

    def unregister(self, tool_id: str) -> None:
        """Unregister a tool."""
        pass
```

---

## Provenance Tracking

### Provenance Record

```python
class ProvenanceRecord(BaseModel):
    """Provenance record for audit trail."""

    record_id: str = Field(..., description="Unique record identifier")
    timestamp: datetime = Field(..., description="Record timestamp")
    agent_id: str = Field(..., description="Agent identifier")
    tool_id: Optional[str] = Field(None, description="Tool identifier")
    input_hash: str = Field(..., description="SHA-256 hash of input")
    output_hash: str = Field(..., description="SHA-256 hash of output")
    formula: Optional[str] = Field(None, description="Formula used")
    standards: List[str] = Field(default_factory=list, description="Standards applied")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
```

### Provenance Tracker

```python
class ProvenanceTracker:
    """Track provenance chain with SHA-256 hashing."""

    def track(
        self,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Track provenance for input/output pair.

        Returns:
            SHA-256 hash of provenance record
        """
        pass

    def get_chain(self) -> List[ProvenanceRecord]:
        """Get complete provenance chain."""
        pass

    def get_chain_hash(self) -> str:
        """Get SHA-256 hash of complete chain."""
        pass

    def validate_chain(self) -> bool:
        """Validate integrity of provenance chain."""
        pass
```

---

## Testing Support

### Test Base Class

```python
from greenlang_sdk.testing import SDKAgentTestCase

class TestMyAgent(SDKAgentTestCase):
    """Test case for MyAgent."""

    def setUp(self):
        """Setup test fixtures."""
        self.agent = MyAgent(pack_path=self.test_pack_path)

    def test_calculation(self):
        """Test emission calculation."""
        input_data = MyInput(amount=1000, fuel_type="natural_gas")
        result = self.agent.run(input_data)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.data.emissions)
        self.assertGreater(result.data.emissions, 0)

    def test_provenance_tracking(self):
        """Test provenance tracking."""
        input_data = MyInput(amount=1000, fuel_type="natural_gas")
        result = self.agent.run(input_data)

        self.assertIn("provenance_hash", result.metadata)
        self.assertTrue(self.agent.validate_provenance())

    def test_tool_execution(self):
        """Test tool execution."""
        tool_result = self.agent.use_tool(
            "calculator",
            {"value": 100, "factor": 2.5}
        )

        self.assertEqual(tool_result.data["result"], 250)
        self.assertTrue(tool_result.deterministic)
```

---

## Implementation Examples

### Example 1: Simple Calculator Agent

```python
from greenlang_sdk.base import SDKAgentBase
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    amount: float = Field(..., ge=0)
    fuel_type: str

class CalculatorOutput(BaseModel):
    emissions_kg_co2e: float

class SimpleCalculatorAgent(SDKAgentBase[CalculatorInput, CalculatorOutput]):
    """Simple emissions calculator agent."""

    def execute_impl(
        self,
        validated_input: CalculatorInput,
        context: AgentExecutionContext
    ) -> CalculatorOutput:
        # Get emission factor
        ef = self.get_emission_factor(validated_input.fuel_type)

        # Calculate emissions using tool
        result = self.use_tool(
            "emissions_calculator",
            {
                "activity_data": validated_input.amount,
                "emission_factor": ef["value"],
                "unit": "kg"
            }
        )

        return CalculatorOutput(
            emissions_kg_co2e=result.data["co2e_emissions_kg"]
        )
```

### Example 2: Validation Agent

```python
class ValidationInput(BaseModel):
    data: Dict[str, Any]
    schema: str

class ValidationOutput(BaseModel):
    is_valid: bool
    errors: List[str]

class DataValidationAgent(SDKAgentBase[ValidationInput, ValidationOutput]):
    """Data validation agent."""

    def execute_impl(
        self,
        validated_input: ValidationInput,
        context: AgentExecutionContext
    ) -> ValidationOutput:
        # Validate data against schema
        result = self.use_tool(
            "data_validator",
            {
                "data": validated_input.data,
                "schema": validated_input.schema
            }
        )

        return ValidationOutput(
            is_valid=result.data["is_valid"],
            errors=result.data.get("errors", [])
        )
```

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-03
**Author**: GL-BackendDeveloper
**Status**: Specification - Ready for Implementation
