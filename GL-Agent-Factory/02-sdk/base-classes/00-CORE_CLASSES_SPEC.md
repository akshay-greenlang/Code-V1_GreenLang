# Core SDK Classes - Implementation Specification

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** Specification

## Executive Summary

This document specifies the core SDK classes that extend AgentSpecV2Base with domain-specific capabilities. These classes provide production-ready base implementations for Calculator agents, Regulatory agents, Reporting agents, and system integration agents.

---

## Table of Contents

1. [Class Hierarchy](#class-hierarchy)
2. [CalculatorAgentBase](#calculatoragentbase)
3. [ValidatorAgentBase](#validatoragentbase)
4. [RegulatoryAgentBase](#regulatoryagentbase)
5. [ReportingAgentBase](#reportingagentbase)
6. [IntegrationAgentBase](#integrationagentbase)
7. [OrchestratorAgentBase](#orchestratoragentbase)
8. [Implementation Plan](#implementation-plan)

---

## Class Hierarchy

### Full Inheritance Tree

```
AgentSpecV2Base[InT, OutT]                    # Core framework
    ↓
SDKAgentBase[InT, OutT]                       # SDK enhancements
    ├── CalculatorAgentBase[InT, OutT]       # Zero-hallucination calculations
    ├── ValidatorAgentBase[InT, OutT]        # Data validation
    ├── RegulatoryAgentBase[InT, OutT]       # Regulatory compliance
    ├── ReportingAgentBase[InT, OutT]        # Report generation
    ├── IntegrationAgentBase[InT, OutT]      # System integration
    └── OrchestratorAgentBase[InT, OutT]     # Multi-agent orchestration
```

### Responsibility Matrix

| Base Class | Primary Responsibility | Tools Used | Output Type |
|------------|----------------------|------------|-------------|
| **CalculatorAgentBase** | Deterministic calculations | Calculator tools, EF database | Numeric results |
| **ValidatorAgentBase** | Schema and constraint validation | Validation tools | Validation results |
| **RegulatoryAgentBase** | Compliance checking and mapping | Framework mappers | Compliance records |
| **ReportingAgentBase** | Report generation | Template engines | Reports (PDF, Excel, JSON) |
| **IntegrationAgentBase** | External system integration | Connectors (SCADA, ERP) | Synced data |
| **OrchestratorAgentBase** | Multi-agent coordination | Agent registry | Aggregated results |

---

## CalculatorAgentBase

### Purpose

Base class for agents that perform deterministic calculations with zero-hallucination guarantee. All numeric calculations MUST go through validated tools.

### Class Definition

```python
"""
Calculator Agent Base
=====================

Zero-hallucination calculation agent with comprehensive validation and provenance.
"""

from typing import Dict, Any, List, Optional
from abc import abstractmethod
from pydantic import BaseModel, Field
from greenlang_sdk.base import SDKAgentBase
from greenlang_sdk.tools import EmissionsCalculator, UnitConverter
from greenlang_sdk.models import CalculationResult, ProvenanceRecord


class CalculatorInput(BaseModel):
    """Base input model for calculator agents."""

    activity_data: float = Field(..., ge=0, description="Activity data value")
    unit: str = Field(..., description="Unit of measurement")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CalculatorOutput(BaseModel):
    """Base output model for calculator agents."""

    result: float = Field(..., description="Calculation result")
    unit: str = Field(..., description="Result unit")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculation_method: str = Field(..., description="Method/formula used")
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Confidence/accuracy score")


class CalculatorAgentBase(SDKAgentBase[CalculatorInput, CalculatorOutput]):
    """
    Base class for zero-hallucination calculation agents.

    This class enforces:
    - All calculations through deterministic tools
    - Complete provenance tracking
    - Unit validation and conversion
    - Error propagation and uncertainty quantification

    Subclasses must implement:
    - get_calculation_parameters(): Prepare tool parameters
    - validate_calculation_result(): Validate tool output

    Example:
        >>> class EmissionsCalculator(CalculatorAgentBase):
        ...     def get_calculation_parameters(self, input: CalculatorInput) -> Dict:
        ...         ef = self.get_emission_factor(input.metadata["fuel_type"])
        ...         return {
        ...             "activity_data": input.activity_data,
        ...             "emission_factor": ef["value"],
        ...             "unit": input.unit
        ...         }
        ...
        ...     def validate_calculation_result(self, result: Dict) -> bool:
        ...         return result["emissions"] >= 0
    """

    def __init__(self, **kwargs):
        """Initialize calculator agent with calculation tools."""
        super().__init__(domain="calculation", **kwargs)

        # Register calculation tools
        self.register_tool("emissions_calculator", EmissionsCalculator())
        self.register_tool("unit_converter", UnitConverter())

        # Calculation history for audit
        self.calculation_history: List[CalculationResult] = []

    def execute_impl(
        self,
        validated_input: CalculatorInput,
        context: AgentExecutionContext
    ) -> CalculatorOutput:
        """
        Execute calculation with full provenance tracking.

        This method:
        1. Validates units
        2. Gets calculation parameters from subclass
        3. Executes calculation tool
        4. Validates result
        5. Tracks provenance
        """
        # Step 1: Validate units
        self._validate_units(validated_input.unit)

        # Step 2: Get calculation parameters (from subclass)
        calc_params = self.get_calculation_parameters(validated_input)

        # Step 3: Execute calculation tool (ZERO HALLUCINATION)
        tool_result = self.use_tool(
            tool_id=self.get_calculator_tool_id(),
            parameters=calc_params,
            track_provenance=True
        )

        # Step 4: Validate result (subclass validation)
        if not self.validate_calculation_result(tool_result.data):
            raise CalculationError(
                f"Calculation validation failed: {tool_result.data}"
            )

        # Step 5: Extract result
        result_value = self._extract_result_value(tool_result.data)
        result_unit = self._extract_result_unit(tool_result.data)

        # Step 6: Calculate accuracy/uncertainty
        accuracy = self._calculate_accuracy(tool_result)

        # Step 7: Track in history
        calc_result = CalculationResult(
            input_data=validated_input.dict(),
            output_value=result_value,
            output_unit=result_unit,
            tool_id=self.get_calculator_tool_id(),
            provenance_hash=tool_result.provenance["output_hash"],
            timestamp=datetime.now()
        )
        self.calculation_history.append(calc_result)

        # Step 8: Create output
        return CalculatorOutput(
            result=result_value,
            unit=result_unit,
            provenance_hash=tool_result.provenance["output_hash"],
            calculation_method=tool_result.provenance.get("formula", "unknown"),
            accuracy=accuracy
        )

    # =========================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def get_calculation_parameters(self, input: CalculatorInput) -> Dict[str, Any]:
        """
        Prepare parameters for calculation tool.

        Args:
            input: Validated input data

        Returns:
            Parameters dictionary for calculator tool

        Example:
            >>> def get_calculation_parameters(self, input):
            ...     ef = self.get_emission_factor(input.metadata["fuel_type"])
            ...     return {
            ...         "activity_data": input.activity_data,
            ...         "emission_factor": ef["value"]
            ...     }
        """
        pass

    @abstractmethod
    def validate_calculation_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate calculation tool output.

        Args:
            result: Tool output data

        Returns:
            True if valid, False otherwise

        Example:
            >>> def validate_calculation_result(self, result):
            ...     # Check non-negative emissions
            ...     return result.get("emissions", -1) >= 0
        """
        pass

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def get_calculator_tool_id(self) -> str:
        """
        Get calculator tool ID (override if using different tool).

        Returns:
            Tool identifier
        """
        return "emissions_calculator"

    def _validate_units(self, unit: str) -> None:
        """Validate unit against climate units whitelist."""
        valid_units = self._get_valid_units()
        if unit not in valid_units:
            raise ValidationError(
                f"Invalid unit '{unit}'. Must be one of: {valid_units}"
            )

    def _get_valid_units(self) -> List[str]:
        """Get list of valid units for this calculator."""
        return [
            "kg", "t", "MT", "lb",  # Mass
            "kWh", "MWh", "GJ", "BTU", "mmBTU",  # Energy
            "m3", "L", "gal",  # Volume
            "kgCO2e", "tCO2e", "MTCO2e"  # Emissions
        ]

    def _extract_result_value(self, tool_data: Dict) -> float:
        """Extract numeric result from tool output."""
        # Try common field names
        for key in ["result", "emissions", "value", "output"]:
            if key in tool_data:
                return float(tool_data[key])

        raise ValueError("Could not extract result value from tool output")

    def _extract_result_unit(self, tool_data: Dict) -> str:
        """Extract result unit from tool output."""
        return tool_data.get("unit", "unknown")

    def _calculate_accuracy(self, tool_result: ToolResult) -> Optional[float]:
        """Calculate accuracy/confidence score from tool result."""
        # Extract uncertainty information if available
        uncertainty = tool_result.provenance.get("uncertainty")
        if uncertainty:
            return 1.0 - uncertainty
        return None

    def get_calculation_history(self) -> List[CalculationResult]:
        """Get complete calculation history for this agent."""
        return self.calculation_history
```

### Usage Example

```python
from greenlang_sdk.base import CalculatorAgentBase

class CarbonEmissionsCalculator(CalculatorAgentBase):
    """Calculate carbon emissions from fuel consumption."""

    def get_calculation_parameters(self, input: CalculatorInput) -> Dict:
        # Get emission factor from database
        fuel_type = input.metadata.get("fuel_type")
        region = input.metadata.get("region", "US")

        ef = self.get_emission_factor(
            material_id=fuel_type,
            region=region
        )

        return {
            "activity_data": input.activity_data,
            "emission_factor": ef["value"],
            "unit": input.unit,
            "gwp_set": "AR6"
        }

    def validate_calculation_result(self, result: Dict) -> bool:
        # Validate non-negative emissions
        emissions = result.get("co2e_emissions_kg", -1)
        return emissions >= 0

# Use agent
agent = CarbonEmissionsCalculator()
result = agent.run(CalculatorInput(
    activity_data=1000,
    unit="kg",
    metadata={"fuel_type": "natural_gas", "region": "US"}
))

print(f"Emissions: {result.data.result} {result.data.unit}")
print(f"Provenance: {result.data.provenance_hash}")
```

---

## ValidatorAgentBase

### Purpose

Base class for agents that validate data against schemas, constraints, and business rules.

### Class Definition

```python
"""
Validator Agent Base
====================

Data validation agent with schema checking, constraint validation, and business rules.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from greenlang_sdk.base import SDKAgentBase
from greenlang_sdk.validation import SchemaValidator, ConstraintValidator


class ValidationInput(BaseModel):
    """Input for validation agents."""

    data: Dict[str, Any] = Field(..., description="Data to validate")
    schema_id: Optional[str] = Field(None, description="Schema identifier")
    validation_rules: List[str] = Field(default_factory=list, description="Validation rule IDs")


class ValidationOutput(BaseModel):
    """Output from validation agents."""

    is_valid: bool = Field(..., description="Whether data is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    validated_data: Optional[Dict] = Field(None, description="Validated/transformed data")


class ValidatorAgentBase(SDKAgentBase[ValidationInput, ValidationOutput]):
    """
    Base class for data validation agents.

    This class provides:
    - Schema validation (JSON Schema, Pydantic models)
    - Constraint validation (ranges, enums, patterns)
    - Business rule validation (custom rules)
    - Data transformation and normalization

    Example:
        >>> class EmissionsDataValidator(ValidatorAgentBase):
        ...     def get_validation_rules(self, input: ValidationInput) -> List[ValidationRule]:
        ...         return [
        ...             RangeRule("emissions", min=0),
        ...             EnumRule("fuel_type", allowed=["gas", "coal", "oil"]),
        ...             CustomRule("ghg_protocol_compliance", self.check_ghg_compliance)
        ...         ]
    """

    def __init__(self, **kwargs):
        super().__init__(domain="validation", **kwargs)

        # Register validation tools
        self.schema_validator = SchemaValidator()
        self.constraint_validator = ConstraintValidator()

    def execute_impl(
        self,
        validated_input: ValidationInput,
        context: AgentExecutionContext
    ) -> ValidationOutput:
        """Execute validation checks."""
        errors = []
        warnings = []
        validated_data = validated_input.data.copy()

        # Step 1: Schema validation
        if validated_input.schema_id:
            schema_result = self._validate_schema(
                validated_input.data,
                validated_input.schema_id
            )
            errors.extend(schema_result.errors)
            warnings.extend(schema_result.warnings)

        # Step 2: Constraint validation
        constraint_result = self._validate_constraints(validated_input.data)
        errors.extend(constraint_result.errors)
        warnings.extend(constraint_result.warnings)

        # Step 3: Business rule validation (from subclass)
        rules = self.get_validation_rules(validated_input)
        for rule in rules:
            rule_result = rule.validate(validated_input.data)
            errors.extend(rule_result.errors)
            warnings.extend(rule_result.warnings)

        # Step 4: Data transformation (if validation passed)
        if not errors:
            validated_data = self.transform_data(validated_input.data)

        return ValidationOutput(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validated_data=validated_data if not errors else None
        )

    @abstractmethod
    def get_validation_rules(self, input: ValidationInput) -> List[ValidationRule]:
        """Get validation rules for this input."""
        pass

    def transform_data(self, data: Dict) -> Dict:
        """Transform/normalize data (override if needed)."""
        return data
```

---

## RegulatoryAgentBase

### Purpose

Base class for agents that ensure regulatory compliance, map data to frameworks, and generate audit trails.

### Class Definition

```python
"""
Regulatory Agent Base
=====================

Regulatory compliance agent with framework mapping and audit trail generation.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from greenlang_sdk.base import SDKAgentBase


class RegulatoryInput(BaseModel):
    """Input for regulatory agents."""

    data: Dict[str, Any] = Field(..., description="Data to validate")
    frameworks: List[str] = Field(..., description="Target frameworks (GRI, SASB, TCFD, CDP)")
    jurisdiction: str = Field(..., description="Regulatory jurisdiction")


class RegulatoryOutput(BaseModel):
    """Output from regulatory agents."""

    is_compliant: bool = Field(..., description="Whether data is compliant")
    framework_mappings: Dict[str, Any] = Field(..., description="Mapped data by framework")
    compliance_gaps: List[str] = Field(default_factory=list, description="Compliance gaps")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    audit_trail: Dict[str, Any] = Field(..., description="Complete audit trail")


class RegulatoryAgentBase(SDKAgentBase[RegulatoryInput, RegulatoryOutput]):
    """
    Base class for regulatory compliance agents.

    Features:
    - Multi-framework mapping (GRI, SASB, TCFD, CDP, ISSB, CSRD)
    - Jurisdiction-specific compliance checking
    - Gap analysis and recommendations
    - Complete audit trail generation
    """

    def __init__(self, **kwargs):
        super().__init__(domain="regulatory", **kwargs)

        # Supported frameworks
        self.supported_frameworks = [
            "GRI_UNIVERSAL", "GRI_305", "GRI_302",
            "SASB", "TCFD", "CDP", "ISSB", "CSRD"
        ]

    def execute_impl(
        self,
        validated_input: RegulatoryInput,
        context: AgentExecutionContext
    ) -> RegulatoryOutput:
        """Execute regulatory compliance check."""

        # Step 1: Validate frameworks
        for framework in validated_input.frameworks:
            if framework not in self.supported_frameworks:
                raise ValueError(f"Unsupported framework: {framework}")

        # Step 2: Map to each framework
        framework_mappings = {}
        for framework in validated_input.frameworks:
            mapping = self.map_to_framework(
                validated_input.data,
                framework
            )
            framework_mappings[framework] = mapping

        # Step 3: Check compliance
        compliance_check = self._check_compliance(
            validated_input.data,
            validated_input.jurisdiction
        )

        # Step 4: Identify gaps
        gaps = self._identify_gaps(framework_mappings, validated_input.frameworks)

        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations(gaps)

        # Step 6: Create audit trail
        audit_trail = self._create_audit_trail(
            validated_input,
            framework_mappings,
            compliance_check
        )

        return RegulatoryOutput(
            is_compliant=compliance_check["is_compliant"],
            framework_mappings=framework_mappings,
            compliance_gaps=gaps,
            recommendations=recommendations,
            audit_trail=audit_trail
        )
```

---

## ReportingAgentBase

### Purpose

Base class for agents that generate reports in various formats (PDF, Excel, JSON).

### Class Definition

```python
"""
Reporting Agent Base
====================

Report generation agent with template rendering and multi-format output.
"""

from typing import Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field
from greenlang_sdk.base import SDKAgentBase


class ReportFormat(str, Enum):
    """Supported report formats."""
    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    HTML = "html"
    CSV = "csv"


class ReportingInput(BaseModel):
    """Input for reporting agents."""

    data: Dict[str, Any] = Field(..., description="Data to report")
    template_id: str = Field(..., description="Report template identifier")
    format: ReportFormat = Field(ReportFormat.PDF, description="Output format")
    include_charts: bool = Field(True, description="Include visualizations")


class ReportingOutput(BaseModel):
    """Output from reporting agents."""

    report_content: Any = Field(..., description="Report content (bytes or dict)")
    report_format: ReportFormat = Field(..., description="Report format")
    report_size_bytes: int = Field(..., description="Report size")
    generation_time_ms: float = Field(..., description="Generation time")


class ReportingAgentBase(SDKAgentBase[ReportingInput, ReportingOutput]):
    """
    Base class for report generation agents.

    Features:
    - Multi-format output (PDF, Excel, JSON, HTML, CSV)
    - Template-based rendering
    - Chart/visualization generation
    - Executive summaries
    """

    def execute_impl(
        self,
        validated_input: ReportingInput,
        context: AgentExecutionContext
    ) -> ReportingOutput:
        """Generate report."""
        start_time = time.time()

        # Step 1: Load template
        template = self._load_template(validated_input.template_id)

        # Step 2: Prepare data
        report_data = self.prepare_report_data(validated_input.data)

        # Step 3: Generate charts
        charts = []
        if validated_input.include_charts:
            charts = self.generate_charts(report_data)

        # Step 4: Render report
        report_content = self._render_report(
            template=template,
            data=report_data,
            charts=charts,
            format=validated_input.format
        )

        generation_time = (time.time() - start_time) * 1000

        return ReportingOutput(
            report_content=report_content,
            report_format=validated_input.format,
            report_size_bytes=len(report_content) if isinstance(report_content, bytes) else 0,
            generation_time_ms=generation_time
        )

    @abstractmethod
    def prepare_report_data(self, data: Dict) -> Dict:
        """Prepare data for report rendering."""
        pass

    def generate_charts(self, data: Dict) -> List[Any]:
        """Generate charts/visualizations (override if needed)."""
        return []
```

---

## IntegrationAgentBase

### Purpose

Base class for agents that integrate with external systems (SCADA, ERP, CMMS).

### Class Definition

```python
"""
Integration Agent Base
======================

System integration agent with connector management and data synchronization.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from greenlang_sdk.base import SDKAgentBase
from greenlang_sdk.integration import SCADAConnector, ERPConnector


class IntegrationInput(BaseModel):
    """Input for integration agents."""

    operation: str = Field(..., description="Operation (read, write, sync)")
    system_id: str = Field(..., description="External system identifier")
    query: Dict[str, Any] = Field(..., description="Query parameters")


class IntegrationOutput(BaseModel):
    """Output from integration agents."""

    success: bool = Field(..., description="Whether operation succeeded")
    data: Dict[str, Any] = Field(..., description="Retrieved/synced data")
    records_count: int = Field(0, description="Number of records processed")
    sync_timestamp: datetime = Field(..., description="Synchronization timestamp")


class IntegrationAgentBase(SDKAgentBase[IntegrationInput, IntegrationOutput]):
    """
    Base class for system integration agents.

    Features:
    - SCADA/DCS integration (OPC UA, Modbus, MQTT)
    - ERP integration (SAP, Oracle, REST APIs)
    - CMMS integration (Maximo, SAP PM)
    - Data synchronization and caching
    """

    def __init__(self, **kwargs):
        super().__init__(domain="integration", **kwargs)

        # Initialize connectors
        self.connectors = {
            "scada": SCADAConnector(),
            "erp": ERPConnector(),
        }

    def execute_impl(
        self,
        validated_input: IntegrationInput,
        context: AgentExecutionContext
    ) -> IntegrationOutput:
        """Execute integration operation."""

        # Get connector
        connector = self._get_connector(validated_input.system_id)

        # Execute operation
        if validated_input.operation == "read":
            data = connector.read_data(validated_input.query)
        elif validated_input.operation == "write":
            connector.write_data(validated_input.query["data"])
            data = {"status": "written"}
        elif validated_input.operation == "sync":
            data = connector.sync_data(validated_input.query)
        else:
            raise ValueError(f"Unknown operation: {validated_input.operation}")

        return IntegrationOutput(
            success=True,
            data=data,
            records_count=len(data) if isinstance(data, list) else 1,
            sync_timestamp=datetime.now()
        )
```

---

## OrchestratorAgentBase

### Purpose

Base class for agents that orchestrate multiple sub-agents with dependency management.

### Class Definition

```python
"""
Orchestrator Agent Base
=======================

Multi-agent orchestration with dependency management and result aggregation.
"""

from typing import Dict, List, Any
from pydantic import BaseModel, Field
from greenlang_sdk.base import SDKAgentBase


class OrchestratorInput(BaseModel):
    """Input for orchestrator agents."""

    request: Dict[str, Any] = Field(..., description="Orchestration request")
    agent_selection: List[str] = Field(..., description="Agents to execute")


class OrchestratorOutput(BaseModel):
    """Output from orchestrator agents."""

    agent_results: Dict[str, Any] = Field(..., description="Results by agent ID")
    execution_order: List[str] = Field(..., description="Execution order")
    total_execution_time_ms: float = Field(..., description="Total execution time")


class OrchestratorAgentBase(SDKAgentBase[OrchestratorInput, OrchestratorOutput]):
    """
    Base class for multi-agent orchestration.

    Features:
    - Agent registry and discovery
    - Dependency-aware execution
    - Parallel and sequential execution
    - Result aggregation
    """

    def __init__(self, sub_agents: Dict[str, SDKAgentBase], **kwargs):
        super().__init__(domain="orchestration", **kwargs)
        self.sub_agents = sub_agents

    def execute_impl(
        self,
        validated_input: OrchestratorInput,
        context: AgentExecutionContext
    ) -> OrchestratorOutput:
        """Execute orchestration."""
        start_time = time.time()

        # Build execution graph
        execution_order = self._build_execution_graph(validated_input.agent_selection)

        # Execute agents
        results = {}
        for agent_id in execution_order:
            agent = self.sub_agents[agent_id]
            result = agent.run(validated_input.request)
            results[agent_id] = result.data

        total_time = (time.time() - start_time) * 1000

        return OrchestratorOutput(
            agent_results=results,
            execution_order=execution_order,
            total_execution_time_ms=total_time
        )
```

---

## Implementation Plan

### Phase 1: Core Base Classes (Week 1)

**Deliverables:**
- `SDKAgentBase` with tool management
- `CalculatorAgentBase` with zero-hallucination enforcement
- `ValidatorAgentBase` with schema validation
- Unit tests for all classes (85%+ coverage)

### Phase 2: Domain-Specific Classes (Week 2)

**Deliverables:**
- `RegulatoryAgentBase` with framework mapping
- `ReportingAgentBase` with template rendering
- `IntegrationAgentBase` with connector management
- Integration tests for external systems

### Phase 3: Orchestration (Week 3)

**Deliverables:**
- `OrchestratorAgentBase` with dependency management
- Agent graph patterns implementation
- End-to-end orchestration tests

### Phase 4: Documentation & Examples (Week 4)

**Deliverables:**
- API documentation (Sphinx)
- Usage examples for each base class
- Tutorial series
- Migration guide from bare AgentSpec v2

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-03
**Author**: GL-BackendDeveloper
**Status**: Specification - Ready for Implementation
