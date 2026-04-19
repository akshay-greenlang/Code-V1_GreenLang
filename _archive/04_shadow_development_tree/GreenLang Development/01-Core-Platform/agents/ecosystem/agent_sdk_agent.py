# -*- coding: utf-8 -*-
"""
GL-ECO-X-001: Agent SDK Agent
==============================

Provides SDK capabilities for building custom GreenLang agents, including
templates, code generation, validation, and scaffolding.

Capabilities:
    - Agent template management
    - Code scaffolding and generation
    - Agent definition validation
    - Best practices enforcement
    - Boilerplate generation
    - Integration testing support

Zero-Hallucination Guarantees:
    - All code generation uses deterministic templates
    - Complete provenance tracking with SHA-256 hashes
    - No LLM calls in the code generation path
    - All generated code traceable to templates

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class TemplateType(str, Enum):
    """Types of agent templates."""
    BASIC = "basic"
    CALCULATION = "calculation"
    INTEGRATION = "integration"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ORCHESTRATION = "orchestration"


class AgentLayer(str, Enum):
    """Agent layer classification."""
    FOUNDATION = "foundation"
    DATA = "data"
    MRV = "mrv"
    PLANNING = "planning"
    RISK = "risk"
    FINANCE = "finance"
    PROCUREMENT = "procurement"
    POLICY = "policy"
    REPORTING = "reporting"
    OPERATIONS = "operations"
    ECOSYSTEM = "ecosystem"


class ValidationSeverity(str, Enum):
    """Severity of validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# Pydantic Models
# =============================================================================

class AgentCapability(BaseModel):
    """Definition of an agent capability."""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    input_types: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)


class AgentDefinition(BaseModel):
    """Definition for a new agent."""
    agent_id: str = Field(..., description="Agent ID (e.g., GL-MRV-X-001)")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    version: str = Field(default="1.0.0", description="Version")
    layer: AgentLayer = Field(..., description="Agent layer")
    template_type: TemplateType = Field(default=TemplateType.BASIC)

    # Capabilities
    capabilities: List[AgentCapability] = Field(default_factory=list)

    # Input/Output
    input_model_name: str = Field(default="AgentInput")
    output_model_name: str = Field(default="AgentOutput")
    input_fields: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    output_fields: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Configuration
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Author info
    author: Optional[str] = Field(None)
    email: Optional[str] = Field(None)

    @field_validator('agent_id')
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Validate agent ID format."""
        import re
        pattern = r'^GL-[A-Z]+-[A-Z]-\d{3}$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid agent ID format: {v}. Expected: GL-LAYER-X-NNN")
        return v


class AgentTemplate(BaseModel):
    """An agent template for code generation."""
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Template name")
    template_type: TemplateType = Field(..., description="Template type")
    description: str = Field(..., description="Template description")
    template_code: str = Field(..., description="Template code")
    required_imports: List[str] = Field(default_factory=list)
    placeholders: List[str] = Field(default_factory=list)
    version: str = Field(default="1.0.0")


class ValidationIssue(BaseModel):
    """A validation issue found in agent definition."""
    severity: ValidationSeverity = Field(..., description="Issue severity")
    code: str = Field(..., description="Issue code")
    message: str = Field(..., description="Issue message")
    location: Optional[str] = Field(None, description="Location in definition")
    suggestion: Optional[str] = Field(None, description="Fix suggestion")


class ValidationResult(BaseModel):
    """Result of agent definition validation."""
    valid: bool = Field(..., description="Whether definition is valid")
    issues: List[ValidationIssue] = Field(default_factory=list)
    error_count: int = Field(default=0)
    warning_count: int = Field(default=0)
    info_count: int = Field(default=0)


class CodeGenerationResult(BaseModel):
    """Result of code generation."""
    success: bool = Field(..., description="Whether generation succeeded")
    agent_code: str = Field(default="", description="Generated agent code")
    test_code: str = Field(default="", description="Generated test code")
    init_code: str = Field(default="", description="Generated __init__.py additions")
    file_path: str = Field(default="", description="Suggested file path")
    provenance_hash: str = Field(default="", description="Code provenance hash")


class AgentSDKInput(BaseModel):
    """Input for the Agent SDK Agent."""
    operation: str = Field(..., description="Operation to perform")
    agent_definition: Optional[AgentDefinition] = Field(None)
    template: Optional[AgentTemplate] = Field(None)
    template_type: Optional[TemplateType] = Field(None)
    code: Optional[str] = Field(None, description="Code to validate")

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation is supported."""
        valid_ops = {
            'generate_agent', 'validate_definition', 'get_templates',
            'add_template', 'scaffold_project', 'generate_tests',
            'validate_code', 'get_best_practices'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class AgentSDKOutput(BaseModel):
    """Output from the Agent SDK Agent."""
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation performed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# Default Templates
# =============================================================================

BASIC_AGENT_TEMPLATE = '''# -*- coding: utf-8 -*-
"""
{agent_id}: {agent_name}
{'=' * (len('{agent_id}: {agent_name}'))}

{description}

Zero-Hallucination Guarantees:
    - All calculations use deterministic formulas
    - Complete provenance tracking with SHA-256 hashes
    - No LLM calls in the calculation path

Author: {author}
Version: {version}
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class {input_model_name}(BaseModel):
    """Input for {agent_name}."""
    operation: str = Field(..., description="Operation to perform")
{input_fields}

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        valid_ops = {{'process', 'get_statistics'}}
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {{valid_ops}}")
        return v


class {output_model_name}(BaseModel):
    """Output from {agent_name}."""
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation performed")
    data: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


class {class_name}(BaseAgent):
    """
    {agent_id}: {agent_name}

    {description}

    Usage:
        agent = {class_name}()
        result = agent.run({{"operation": "process", ...}})
    """

    AGENT_ID = "{agent_id}"
    AGENT_NAME = "{agent_name}"
    VERSION = "{version}"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="{description}",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {{self.AGENT_ID}}: {{self.AGENT_NAME}}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            agent_input = {input_model_name}(**input_data)
            operation = agent_input.operation

            result_data = self._route_operation(agent_input)

            provenance_hash = self._compute_provenance_hash(input_data, result_data)
            processing_time_ms = (time.time() - start_time) * 1000

            output = {output_model_name}(
                success=True,
                operation=operation,
                data=result_data,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Operation failed: {{e}}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _route_operation(self, agent_input: {input_model_name}) -> Dict[str, Any]:
        operation = agent_input.operation

        if operation == "process":
            return self._handle_process(agent_input)
        elif operation == "get_statistics":
            return self._handle_get_statistics()
        else:
            raise ValueError(f"Unknown operation: {{operation}}")

    def _handle_process(self, agent_input: {input_model_name}) -> Dict[str, Any]:
        """Process operation - implement your logic here."""
        # TODO: Implement processing logic
        return {{"processed": True}}

    def _handle_get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {{"agent_id": self.AGENT_ID, "version": self.VERSION}}

    def _compute_provenance_hash(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> str:
        provenance_str = json.dumps(
            {{"input": input_data, "output": output_data}},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]
'''


TEST_TEMPLATE = '''# -*- coding: utf-8 -*-
"""Tests for {agent_id}: {agent_name}"""

import pytest
from datetime import datetime

from {module_path} import {class_name}, {input_model_name}, {output_model_name}


class Test{class_name}:
    """Test cases for {class_name}."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return {class_name}()

    def test_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "{agent_id}"
        assert agent.AGENT_NAME == "{agent_name}"

    def test_process_operation(self, agent):
        """Test process operation."""
        result = agent.run({{"operation": "process"}})
        assert result.success is True
        assert "data" in result.data

    def test_get_statistics(self, agent):
        """Test statistics operation."""
        result = agent.run({{"operation": "get_statistics"}})
        assert result.success is True
        assert result.data["data"]["agent_id"] == "{agent_id}"

    def test_invalid_operation(self, agent):
        """Test handling of invalid operation."""
        result = agent.run({{"operation": "invalid"}})
        assert result.success is False

    def test_provenance_hash(self, agent):
        """Test provenance hash is generated."""
        result = agent.run({{"operation": "process"}})
        assert "provenance_hash" in result.data
        assert len(result.data["provenance_hash"]) == 16
'''


# =============================================================================
# Agent SDK Agent Implementation
# =============================================================================

class AgentSDKAgent(BaseAgent):
    """
    GL-ECO-X-001: Agent SDK Agent

    Provides SDK capabilities for building custom GreenLang agents.

    Usage:
        sdk = AgentSDKAgent()

        # Generate agent code
        result = sdk.run({
            "operation": "generate_agent",
            "agent_definition": {
                "agent_id": "GL-MRV-X-001",
                "name": "My Custom Agent",
                "description": "Does something useful",
                "layer": "mrv"
            }
        })
    """

    AGENT_ID = "GL-ECO-X-001"
    AGENT_NAME = "Agent SDK Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="SDK for building custom GreenLang agents",
                version=self.VERSION,
            )
        super().__init__(config)

        # Load default templates
        self._templates: Dict[str, AgentTemplate] = {}
        self._load_default_templates()

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def _load_default_templates(self):
        """Load default agent templates."""
        basic_template = AgentTemplate(
            name="Basic Agent",
            template_type=TemplateType.BASIC,
            description="Basic agent template with standard structure",
            template_code=BASIC_AGENT_TEMPLATE,
            required_imports=[
                "hashlib", "json", "logging", "time",
                "datetime", "typing", "pydantic",
            ],
            placeholders=[
                "agent_id", "agent_name", "description", "version",
                "author", "class_name", "input_model_name", "output_model_name",
            ],
        )
        self._templates[basic_template.template_id] = basic_template

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            sdk_input = AgentSDKInput(**input_data)
            operation = sdk_input.operation

            result_data = self._route_operation(sdk_input)

            provenance_hash = self._compute_provenance_hash(input_data, result_data)
            processing_time_ms = (time.time() - start_time) * 1000

            output = AgentSDKOutput(
                success=True,
                operation=operation,
                data=result_data,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"SDK operation failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _route_operation(self, sdk_input: AgentSDKInput) -> Dict[str, Any]:
        operation = sdk_input.operation

        if operation == "generate_agent":
            return self._handle_generate_agent(sdk_input.agent_definition)
        elif operation == "validate_definition":
            return self._handle_validate_definition(sdk_input.agent_definition)
        elif operation == "get_templates":
            return self._handle_get_templates(sdk_input.template_type)
        elif operation == "add_template":
            return self._handle_add_template(sdk_input.template)
        elif operation == "scaffold_project":
            return self._handle_scaffold_project(sdk_input.agent_definition)
        elif operation == "generate_tests":
            return self._handle_generate_tests(sdk_input.agent_definition)
        elif operation == "validate_code":
            return self._handle_validate_code(sdk_input.code)
        elif operation == "get_best_practices":
            return self._handle_get_best_practices()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _handle_generate_agent(
        self, definition: Optional[AgentDefinition]
    ) -> Dict[str, Any]:
        """Generate agent code from definition."""
        if not definition:
            return {"error": "agent_definition is required"}

        # Validate first
        validation = self._validate_agent_definition(definition)
        if not validation.valid:
            return {
                "error": "Definition validation failed",
                "validation": validation.model_dump(),
            }

        # Generate code
        result = self._generate_code(definition)

        return {
            "generation_result": result.model_dump(),
            "validation": validation.model_dump(),
        }

    def _generate_code(self, definition: AgentDefinition) -> CodeGenerationResult:
        """Generate code from definition."""
        # Convert agent ID to class name
        parts = definition.agent_id.split("-")
        class_name = "".join(
            word.capitalize() for word in definition.name.split()
        ).replace(" ", "") + "Agent"

        # Format input fields
        input_fields_str = ""
        for field_name, field_config in definition.input_fields.items():
            field_type = field_config.get("type", "Any")
            description = field_config.get("description", "")
            default = field_config.get("default", "...")
            input_fields_str += f'    {field_name}: {field_type} = Field({default}, description="{description}")\n'

        if not input_fields_str:
            input_fields_str = "    # Add input fields here\n    pass"

        # Generate code
        code = BASIC_AGENT_TEMPLATE.format(
            agent_id=definition.agent_id,
            agent_name=definition.name,
            description=definition.description,
            version=definition.version,
            author=definition.author or "GreenLang Team",
            class_name=class_name,
            input_model_name=definition.input_model_name,
            output_model_name=definition.output_model_name,
            input_fields=input_fields_str,
        )

        # Calculate provenance hash
        provenance_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        # Determine file path
        layer = definition.layer.value
        file_name = f"{definition.name.lower().replace(' ', '_')}_agent.py"
        file_path = f"greenlang/agents/{layer}/{file_name}"

        return CodeGenerationResult(
            success=True,
            agent_code=code,
            file_path=file_path,
            provenance_hash=provenance_hash,
        )

    def _handle_validate_definition(
        self, definition: Optional[AgentDefinition]
    ) -> Dict[str, Any]:
        """Validate agent definition."""
        if not definition:
            return {"error": "agent_definition is required"}

        validation = self._validate_agent_definition(definition)
        return validation.model_dump()

    def _validate_agent_definition(
        self, definition: AgentDefinition
    ) -> ValidationResult:
        """Perform validation on agent definition."""
        issues = []

        # Check agent ID format
        if not definition.agent_id.startswith("GL-"):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_AGENT_ID",
                message="Agent ID must start with 'GL-'",
                location="agent_id",
            ))

        # Check description
        if len(definition.description) < 20:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="SHORT_DESCRIPTION",
                message="Description should be at least 20 characters",
                location="description",
                suggestion="Provide a more detailed description",
            ))

        # Check version format
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', definition.version):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="INVALID_VERSION",
                message="Version should follow semver format (X.Y.Z)",
                location="version",
            ))

        # Check capabilities
        if not definition.capabilities:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                code="NO_CAPABILITIES",
                message="No capabilities defined",
                location="capabilities",
                suggestion="Define agent capabilities for better discoverability",
            ))

        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for i in issues if i.severity == ValidationSeverity.INFO)

        return ValidationResult(
            valid=error_count == 0,
            issues=issues,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
        )

    def _handle_get_templates(
        self, template_type: Optional[TemplateType]
    ) -> Dict[str, Any]:
        """Get available templates."""
        templates = list(self._templates.values())

        if template_type:
            templates = [t for t in templates if t.template_type == template_type]

        return {
            "templates": [t.model_dump() for t in templates],
            "count": len(templates),
        }

    def _handle_add_template(
        self, template: Optional[AgentTemplate]
    ) -> Dict[str, Any]:
        """Add a new template."""
        if not template:
            return {"error": "template is required"}

        self._templates[template.template_id] = template

        return {
            "template_id": template.template_id,
            "added": True,
        }

    def _handle_scaffold_project(
        self, definition: Optional[AgentDefinition]
    ) -> Dict[str, Any]:
        """Scaffold a complete agent project."""
        if not definition:
            return {"error": "agent_definition is required"}

        # Generate all components
        code_result = self._generate_code(definition)
        test_result = self._generate_test_code(definition)

        return {
            "agent_code": code_result.agent_code,
            "test_code": test_result,
            "file_structure": {
                "agent_file": code_result.file_path,
                "test_file": code_result.file_path.replace(".py", "_test.py").replace(
                    "greenlang/agents", "tests/agents"
                ),
            },
        }

    def _handle_generate_tests(
        self, definition: Optional[AgentDefinition]
    ) -> Dict[str, Any]:
        """Generate test code for an agent."""
        if not definition:
            return {"error": "agent_definition is required"}

        test_code = self._generate_test_code(definition)

        return {"test_code": test_code}

    def _generate_test_code(self, definition: AgentDefinition) -> str:
        """Generate test code."""
        class_name = "".join(
            word.capitalize() for word in definition.name.split()
        ).replace(" ", "") + "Agent"

        layer = definition.layer.value
        module_name = f"{definition.name.lower().replace(' ', '_')}_agent"
        module_path = f"greenlang.agents.{layer}.{module_name}"

        return TEST_TEMPLATE.format(
            agent_id=definition.agent_id,
            agent_name=definition.name,
            class_name=class_name,
            module_path=module_path,
            input_model_name=definition.input_model_name,
            output_model_name=definition.output_model_name,
        )

    def _handle_validate_code(self, code: Optional[str]) -> Dict[str, Any]:
        """Validate Python code syntax."""
        if not code:
            return {"error": "code is required"}

        try:
            compile(code, "<string>", "exec")
            return {
                "valid": True,
                "message": "Code syntax is valid",
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
            }

    def _handle_get_best_practices(self) -> Dict[str, Any]:
        """Get agent development best practices."""
        return {
            "best_practices": [
                {
                    "category": "Structure",
                    "practices": [
                        "Extend BaseAgent for consistent lifecycle management",
                        "Use Pydantic models for input/output validation",
                        "Include comprehensive docstrings",
                    ],
                },
                {
                    "category": "Zero-Hallucination",
                    "practices": [
                        "Never use LLM for numeric calculations",
                        "Use deterministic formulas from databases/YAML",
                        "Track provenance with SHA-256 hashes",
                    ],
                },
                {
                    "category": "Testing",
                    "practices": [
                        "Achieve 85%+ test coverage",
                        "Test all operations",
                        "Include integration tests",
                    ],
                },
                {
                    "category": "Performance",
                    "practices": [
                        "Cache expensive lookups",
                        "Use batch processing for large datasets",
                        "Track processing time",
                    ],
                },
            ],
        }

    def _compute_provenance_hash(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> str:
        provenance_str = json.dumps(
            {"input": input_data, "output": output_data},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]
