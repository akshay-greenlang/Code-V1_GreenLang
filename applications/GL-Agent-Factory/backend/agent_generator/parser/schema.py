"""
Schema Module for Agent Generator

This module defines Pydantic models that represent the structure of pack.yaml
AgentSpec files. These models provide:
- Type validation for all spec fields
- Default values where appropriate
- Documentation for each field
- Serialization/deserialization support

The schema mirrors the pack.yaml structure used in 08-regulatory-agents/.

Example:
    >>> from backend.agent_generator.parser.schema import AgentSpec, PackMeta
    >>>
    >>> spec = AgentSpec(
    ...     pack=PackMeta(
    ...         id="gl-eudr-compliance-v1",
    ...         name="EUDR Deforestation Compliance Agent",
    ...         description="Validates commodities against EUDR",
    ...         version="1.0.0",
    ...     ),
    ...     agents=[...],
    ... )
"""

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


# =============================================================================
# Enumerations
# =============================================================================


class AgentType(str, Enum):
    """
    Agent classification types.

    These types determine the validation rules and generation patterns:
    - due-diligence-validator: Compliance checking agents
    - deterministic-calculator: Zero-hallucination calculation agents
    - ml-classifier: ML-based classification agents (can use LLM)
    - report-generator: Report generation agents (can use LLM for narratives)
    """

    DUE_DILIGENCE_VALIDATOR = "due-diligence-validator"
    DETERMINISTIC_CALCULATOR = "deterministic-calculator"
    ML_CLASSIFIER = "ml-classifier"
    REPORT_GENERATOR = "report-generator"


class ToolType(str, Enum):
    """
    Tool classification types.

    These types determine how tool wrappers are generated:
    - deterministic: Pure functions with lookup tables (no LLM)
    - external_api: HTTP client wrappers for external services
    - ml_model: ML inference wrappers (can use LLM)
    """

    DETERMINISTIC = "deterministic"
    EXTERNAL_API = "external_api"
    ML_MODEL = "ml_model"


class Severity(str, Enum):
    """Validation rule severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# Input/Output Definitions
# =============================================================================


class InputDef(BaseModel):
    """
    Input field definition for an agent.

    Defines a single input parameter with its type, validation rules,
    and documentation.

    Attributes:
        name: Field name (snake_case recommended)
        type: Data type (string, integer, number, boolean, date, etc.)
        required: Whether the field is required
        default: Default value if not provided
        description: Human-readable description
        validation: Additional validation rules (min, max, pattern, etc.)

    Example:
        >>> input_def = InputDef(
        ...     name="quantity_kg",
        ...     type="number",
        ...     required=True,
        ...     description="Quantity in kilograms",
        ...     validation={"minimum": 0},
        ... )
    """

    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Data type")
    required: bool = Field(default=True, description="Whether field is required")
    default: Optional[Any] = Field(default=None, description="Default value")
    description: str = Field(default="", description="Field description")
    validation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Validation rules (min, max, pattern, enum, etc.)"
    )

    @property
    def constraints(self) -> Dict[str, Any]:
        """Alias for validation dict (for backward compatibility)."""
        return self.validation or {}

    @validator("name")
    def validate_name(cls, v: str) -> str:
        """Validate field name is a valid Python identifier."""
        if not v.isidentifier():
            raise ValueError(f"'{v}' is not a valid Python identifier")
        return v


class OutputDef(BaseModel):
    """
    Output field definition for an agent.

    Defines a single output field with its type and documentation.

    Attributes:
        name: Field name (snake_case recommended)
        type: Data type
        description: Human-readable description

    Example:
        >>> output_def = OutputDef(
        ...     name="risk_level",
        ...     type="string",
        ...     description="Calculated risk level (low, medium, high)",
        ... )
    """

    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Data type")
    description: str = Field(default="", description="Field description")


# =============================================================================
# Tool Definitions
# =============================================================================


class ToolConfig(BaseModel):
    """
    Tool configuration for external APIs.

    Provides connection and rate limiting settings for external API tools.

    Attributes:
        base_url: Base URL for the API
        auth_type: Authentication type (api_key, oauth2, oauth2_mtls)
        rate_limit: Maximum requests per minute
        timeout: Request timeout in seconds
        retry_count: Number of retries on failure

    Example:
        >>> config = ToolConfig(
        ...     base_url="https://api.example.com",
        ...     auth_type="api_key",
        ...     rate_limit=100,
        ...     timeout=30,
        ... )
    """

    base_url: Optional[str] = Field(default=None, description="API base URL")
    auth_type: Optional[str] = Field(default=None, description="Authentication type")
    rate_limit: Optional[int] = Field(default=None, description="Requests per minute")
    timeout: Optional[int] = Field(default=30, description="Timeout in seconds")
    retry_count: Optional[int] = Field(default=3, description="Retry attempts")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Custom headers")


class ToolDef(BaseModel):
    """
    Tool definition from pack.yaml.

    Defines a tool that an agent can use for its calculations or operations.

    Attributes:
        id: Unique tool identifier (snake_case)
        name: Human-readable tool name
        description: Tool description
        type: Tool type (deterministic, external_api, ml_model)
        input_schema: JSON Schema for tool input
        output_schema: JSON Schema for tool output
        config: Optional configuration for external APIs

    Example:
        >>> tool = ToolDef(
        ...     id="country_lookup",
        ...     name="Country Lookup",
        ...     type=ToolType.DETERMINISTIC,
        ...     description="Look up country risk from database",
        ...     input_schema={"type": "object", "properties": {"country_code": {"type": "string"}}},
        ...     output_schema={"type": "object", "properties": {"risk_level": {"type": "string"}}},
        ... )
    """

    id: str = Field(..., description="Tool identifier")
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    type: ToolType = Field(..., description="Tool type")
    input_schema: Dict[str, Any] = Field(..., description="Input JSON Schema")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Output JSON Schema")
    config: Optional[ToolConfig] = Field(default=None, description="API configuration")


# =============================================================================
# Agent Definition
# =============================================================================


class AgentDef(BaseModel):
    """
    Agent definition from pack.yaml.

    Defines a complete agent with its inputs, outputs, and tools.

    Attributes:
        id: Unique agent identifier
        name: Human-readable agent name
        type: Agent type classification
        description: Agent description
        inputs: List of input field definitions
        outputs: List of output field definitions
        tools: List of tool IDs this agent uses

    Example:
        >>> agent = AgentDef(
        ...     id="eudr-validator",
        ...     name="EUDR Compliance Validator",
        ...     type=AgentType.DUE_DILIGENCE_VALIDATOR,
        ...     description="Validates EUDR compliance",
        ...     inputs=[InputDef(name="commodity_type", type="string")],
        ...     outputs=[OutputDef(name="compliance_status", type="string")],
        ...     tools=["country_lookup", "geolocation_validator"],
        ... )
    """

    id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    type: AgentType = Field(..., description="Agent type")
    description: str = Field(..., description="Agent description")
    inputs: List[InputDef] = Field(default_factory=list, description="Input definitions")
    outputs: List[OutputDef] = Field(default_factory=list, description="Output definitions")
    tools: List[str] = Field(default_factory=list, description="Tool IDs used by this agent")

    def get_class_name(self) -> str:
        """Generate PascalCase class name from agent id or name."""
        import re
        # Use id if available, otherwise use name
        source = self.id or self.name
        # Split on hyphens, underscores, and spaces
        parts = re.split(r'[-_\s]+', source)
        # Capitalize each part and join
        return ''.join(word.capitalize() for word in parts if word)

    def get_module_name(self) -> str:
        """Generate snake_case module name from agent id or name."""
        import re
        # Use id if available, otherwise use name
        source = self.id or self.name
        # Convert to snake_case
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', source)
        s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
        return s.replace('-', '_').lower()


# =============================================================================
# Validation Rules
# =============================================================================


class ValidationRule(BaseModel):
    """
    Validation rule definition.

    Defines a validation rule that will be applied to agent inputs.

    Attributes:
        id: Rule identifier
        description: Rule description
        condition: Condition expression (pseudo-code)
        severity: Rule severity (error, warning, info)

    Example:
        >>> rule = ValidationRule(
        ...     id="production_date_after_cutoff",
        ...     description="Production must be after December 31, 2020",
        ...     condition="input.production_date >= '2020-12-31'",
        ...     severity="error",
        ... )
    """

    id: str = Field(..., description="Rule identifier")
    description: str = Field(..., description="Rule description")
    condition: str = Field(..., description="Condition expression")
    severity: str = Field(default="error", description="Severity level")


# =============================================================================
# Golden Tests
# =============================================================================


class GoldenTestCategory(BaseModel):
    """
    Golden test category definition.

    Groups related tests together for organization.

    Attributes:
        name: Category name
        tests: Number of tests in this category
        description: Category description
    """

    name: str = Field(..., description="Category name")
    tests: int = Field(..., ge=0, description="Number of tests")
    description: str = Field(default="", description="Category description")


class GoldenTestSpec(BaseModel):
    """
    Golden tests specification.

    Defines the test suite that should be generated for the agent.

    Attributes:
        count: Total number of tests
        categories: List of test categories

    Example:
        >>> golden_tests = GoldenTestSpec(
        ...     count=200,
        ...     categories=[
        ...         GoldenTestCategory(name="commodity_validation", tests=35),
        ...         GoldenTestCategory(name="edge_cases", tests=15),
        ...     ],
        ... )
    """

    count: int = Field(..., ge=0, description="Total test count")
    categories: List[GoldenTestCategory] = Field(
        default_factory=list,
        description="Test categories"
    )


# =============================================================================
# Metadata
# =============================================================================


class MetadataRegulation(BaseModel):
    """
    Regulatory metadata.

    Provides reference information for the regulation being implemented.

    Attributes:
        name: Regulation name
        reference: Official reference number
        official_journal: Official journal reference
        enforcement_date: Enforcement date
        sme_enforcement_date: SME enforcement date (if different)
        cutoff_date: Cutoff date for compliance
    """

    name: str = Field(..., description="Regulation name")
    reference: str = Field(..., description="Official reference")
    official_journal: Optional[str] = Field(default=None, description="Official journal")
    enforcement_date: Optional[str] = Field(default=None, description="Enforcement date")
    sme_enforcement_date: Optional[str] = Field(default=None, description="SME enforcement date")
    cutoff_date: Optional[str] = Field(default=None, description="Cutoff date")


class CommodityInfo(BaseModel):
    """
    Commodity information.

    Defines a regulated commodity with its CN codes.

    Attributes:
        code: Commodity code
        cn_codes: List of CN codes
        description: Commodity description
    """

    code: str = Field(..., description="Commodity code")
    cn_codes: List[str] = Field(default_factory=list, description="CN codes")
    description: str = Field(default="", description="Commodity description")


class PenaltyInfo(BaseModel):
    """
    Penalty information.

    Defines the penalties for non-compliance.

    Attributes:
        max_fine_percentage: Maximum fine as percentage of turnover
        fine_basis: Basis for calculating fine
        additional_penalties: List of additional penalties
    """

    max_fine_percentage: Optional[float] = Field(default=None, description="Max fine %")
    fine_basis: Optional[str] = Field(default=None, description="Fine basis")
    additional_penalties: List[str] = Field(default_factory=list, description="Additional penalties")


# =============================================================================
# Pack Metadata
# =============================================================================


class PackMeta(BaseModel):
    """
    Pack metadata from pack.yaml.

    Provides identification and categorization information for the agent pack.

    Attributes:
        id: Unique pack identifier
        name: Human-readable pack name
        description: Pack description
        version: Semantic version
        author: Author name
        license: License type
        category: Pack category
        priority: Priority level (P0-CRITICAL, P1-HIGH, etc.)
        deadline: Deadline date string

    Example:
        >>> pack = PackMeta(
        ...     id="gl-eudr-compliance-v1",
        ...     name="EUDR Deforestation Compliance Agent",
        ...     description="Validates commodities against EU EUDR",
        ...     version="1.0.0",
        ...     category="regulatory-compliance",
        ...     priority="P0-CRITICAL",
        ... )
    """

    id: str = Field(..., description="Pack identifier")
    name: str = Field(..., description="Pack name")
    description: str = Field(..., description="Pack description")
    version: str = Field(..., description="Semantic version")
    author: str = Field(default="GreenLang", description="Author")
    license: str = Field(default="Proprietary", description="License")
    category: str = Field(default="", description="Pack category")
    priority: str = Field(default="", description="Priority level")
    deadline: Optional[str] = Field(default=None, description="Deadline date")

    @validator("version")
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        parts = v.split(".")
        if len(parts) < 2:
            raise ValueError("Version must be in format 'X.Y' or 'X.Y.Z'")
        return v


# =============================================================================
# Deployment Configuration
# =============================================================================


class KubernetesResources(BaseModel):
    """Kubernetes resource requests/limits."""

    cpu: Optional[str] = Field(default=None, description="CPU request/limit")
    memory: Optional[str] = Field(default=None, description="Memory request/limit")


class KubernetesConfig(BaseModel):
    """Kubernetes deployment configuration."""

    namespace: Optional[str] = Field(default=None, description="Kubernetes namespace")
    replicas: Optional[Dict[str, int]] = Field(default=None, description="Replica counts")
    resources: Optional[Dict[str, KubernetesResources]] = Field(default=None)


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    prometheus: Optional[Dict[str, Any]] = Field(default=None, description="Prometheus config")
    grafana: Optional[Dict[str, Any]] = Field(default=None, description="Grafana config")
    alerts: Optional[List[Dict[str, Any]]] = Field(default=None, description="Alert definitions")


class DeploymentConfig(BaseModel):
    """Deployment configuration."""

    kubernetes: Optional[KubernetesConfig] = Field(default=None)
    monitoring: Optional[MonitoringConfig] = Field(default=None)


# =============================================================================
# Complete Agent Specification
# =============================================================================


class AgentSpec(BaseModel):
    """
    Complete agent specification from pack.yaml.

    This is the root model that represents an entire pack.yaml file.

    Attributes:
        pack: Pack metadata
        metadata: Additional metadata (regulation, commodities, etc.)
        agents: List of agent definitions
        tools: List of tool definitions
        validation: Validation rules
        golden_tests: Golden test specifications
        deployment: Deployment configuration
        documentation: Documentation configuration

    Example:
        >>> spec = AgentSpec(
        ...     pack=PackMeta(
        ...         id="gl-eudr-compliance-v1",
        ...         name="EUDR Compliance Agent",
        ...         description="...",
        ...         version="1.0.0",
        ...     ),
        ...     agents=[
        ...         AgentDef(
        ...             id="eudr-validator",
        ...             name="EUDR Validator",
        ...             type=AgentType.DUE_DILIGENCE_VALIDATOR,
        ...             description="...",
        ...             inputs=[...],
        ...             outputs=[...],
        ...         )
        ...     ],
        ... )
    """

    pack: PackMeta = Field(..., description="Pack metadata")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    agents: List[AgentDef] = Field(..., description="Agent definitions")
    tools: List[ToolDef] = Field(default_factory=list, description="Tool definitions")
    validation: Optional[Dict[str, Any]] = Field(default=None, description="Validation rules")
    golden_tests: Optional[GoldenTestSpec] = Field(default=None, description="Golden tests")
    deployment: Optional[Dict[str, Any]] = Field(default=None, description="Deployment config")
    documentation: Optional[Dict[str, Any]] = Field(default=None, description="Documentation config")

    def get_agent(self, agent_id: str) -> Optional[AgentDef]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def get_tool(self, tool_id: str) -> Optional[ToolDef]:
        """Get tool by ID."""
        for tool in self.tools:
            if tool.id == tool_id:
                return tool
        return None

    def get_regulation_metadata(self) -> Optional[MetadataRegulation]:
        """Get regulation metadata if present."""
        if "regulation" in self.metadata:
            return MetadataRegulation(**self.metadata["regulation"])
        return None

    def get_commodities(self) -> List[CommodityInfo]:
        """Get commodity information if present."""
        if "commodities" in self.metadata:
            return [CommodityInfo(**c) for c in self.metadata["commodities"]]
        return []

    @property
    def primary_agent(self) -> Optional[AgentDef]:
        """Get the primary (first) agent."""
        return self.agents[0] if self.agents else None

    @property
    def tool_ids(self) -> List[str]:
        """Get all tool IDs."""
        return [tool.id for tool in self.tools]

    @property
    def agent_ids(self) -> List[str]:
        """Get all agent IDs."""
        return [agent.id for agent in self.agents]

    @property
    def spec_hash(self) -> str:
        """Generate a SHA-256 hash of the spec for provenance tracking."""
        import hashlib
        import json
        # Create a deterministic JSON representation
        spec_dict = self.dict()
        json_str = json.dumps(spec_dict, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
