"""
GreenLang Agent Factory LLM Integration
Automated agent generation with LLM-powered code generation
"""

import ast
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    """Types of agents that can be generated"""
    DATA_PROCESSOR = "data_processor"
    API_INTEGRATOR = "api_integrator"
    CALCULATOR = "calculator"
    VALIDATOR = "validator"
    REPORTER = "reporter"
    MONITOR = "monitor"

@dataclass
class AgentSpecification:
    """Specification for agent generation"""
    name: str
    type: AgentType
    description: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    dependencies: List[str]
    business_rules: List[str]
    performance_requirements: Dict[str, Any]
    compliance_requirements: List[str]

class AgentFactoryLLM:
    """
    LLM-powered agent factory for automated code generation
    """

    def __init__(self):
        self.code_generator = CodeGenerationPipeline()
        self.test_generator = TestGenerationPipeline()
        self.doc_generator = DocumentationGenerator()
        self.quality_improver = QualityImprover()
        self.validator = AgentValidator()

    async def create_agent(self, spec: AgentSpecification) -> Dict[str, str]:
        """Create a complete agent from specification"""

        # Step 1: Generate core agent code
        agent_code = await self.code_generator.generate_agent(spec)

        # Step 2: Generate comprehensive tests
        test_code = await self.test_generator.generate_tests(spec, agent_code)

        # Step 3: Generate documentation
        documentation = await self.doc_generator.generate_docs(spec, agent_code)

        # Step 4: Improve code quality
        improved_code = await self.quality_improver.improve(agent_code, spec)

        # Step 5: Validate generated agent
        validation = await self.validator.validate(improved_code, test_code)

        if not validation['passed']:
            # Iterate and fix issues
            improved_code = await self._fix_issues(improved_code, validation['issues'])

        return {
            'agent_code': improved_code,
            'test_code': test_code,
            'documentation': documentation,
            'validation_report': validation
        }

    async def _fix_issues(self, code: str, issues: List[str]) -> str:
        """Fix identified issues in generated code"""

        fix_prompt = f"""Fix the following issues in this Python code:

Issues:
{json.dumps(issues, indent=2)}

Code:
```python
{code}
```

Return only the fixed code without explanations."""

        fixed_code = await self.code_generator.generate(fix_prompt)
        return fixed_code


class CodeGenerationPipeline:
    """
    Pipeline for generating agent code with LLMs
    """

    def __init__(self):
        self.prompts = self._load_code_prompts()

    async def generate_agent(self, spec: AgentSpecification) -> str:
        """Generate agent code from specification"""

        # Select appropriate template based on agent type
        template = self._select_template(spec.type)

        # Generate code structure
        structure_prompt = self._build_structure_prompt(spec, template)
        code_structure = await self._generate_with_llm(structure_prompt)

        # Generate business logic
        logic_prompt = self._build_logic_prompt(spec, code_structure)
        business_logic = await self._generate_with_llm(logic_prompt)

        # Generate error handling
        error_prompt = self._build_error_handling_prompt(spec)
        error_handling = await self._generate_with_llm(error_prompt)

        # Combine components
        complete_code = self._combine_components(
            code_structure,
            business_logic,
            error_handling
        )

        # Validate syntax
        if not self._validate_syntax(complete_code):
            complete_code = await self._fix_syntax(complete_code)

        return complete_code

    def _build_structure_prompt(self, spec: AgentSpecification, template: str) -> str:
        """Build prompt for code structure generation"""

        return f"""Generate Python code structure for a GreenLang agent with these specifications:

Agent Name: {spec.name}
Type: {spec.type.value}
Description: {spec.description}

Inputs:
{json.dumps(spec.inputs, indent=2)}

Outputs:
{json.dumps(spec.outputs, indent=2)}

Dependencies:
{json.dumps(spec.dependencies, indent=2)}

Use this template structure:
{template}

Requirements:
1. Use type hints for all functions
2. Include comprehensive docstrings
3. Follow GreenLang coding standards
4. Implement proper logging
5. Use async/await where appropriate
6. NO HALLUCINATED VALUES - all calculations must use provided formulas

Generate the code structure with placeholder methods for business logic."""

    def _build_logic_prompt(self, spec: AgentSpecification, structure: str) -> str:
        """Build prompt for business logic generation"""

        return f"""Implement the business logic for this GreenLang agent:

Current code structure:
```python
{structure}
```

Business Rules:
{json.dumps(spec.business_rules, indent=2)}

Performance Requirements:
{json.dumps(spec.performance_requirements, indent=2)}

Compliance Requirements:
{json.dumps(spec.compliance_requirements, indent=2)}

CRITICAL REQUIREMENTS:
1. All emissions calculations must use EXACT formulas provided
2. Never generate or estimate numeric values
3. All data must be traceable to source
4. Include validation for all inputs
5. Return results with full provenance

Implement the business logic methods."""

    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _select_template(self, agent_type: AgentType) -> str:
        """Select appropriate code template"""

        templates = {
            AgentType.DATA_PROCESSOR: """
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    data: pd.DataFrame
    metadata: Dict[str, Any]
    provenance: Dict[str, Any]
    warnings: List[str]

class {AgentName}:
    \"\"\"
    {Description}
    \"\"\"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validator = DataValidator()
        self.logger = logger

    async def process(self, input_data: Dict[str, Any]) -> ProcessingResult:
        \"\"\"Main processing method\"\"\"
        # Validate inputs
        # Transform data
        # Apply business rules
        # Return with provenance
        pass
""",
            AgentType.CALCULATOR: """
from typing import Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CalculationResult:
    value: float
    unit: str
    formula_used: str
    factors_used: Dict[str, float]
    confidence: float
    data_tier: str
    provenance: Dict[str, Any]

class {AgentName}:
    \"\"\"
    {Description}

    IMPORTANT: All calculations use deterministic formulas.
    No values are estimated or generated by AI.
    \"\"\"

    def __init__(self):
        self.emission_factors = self._load_emission_factors()
        self.formulas = self._load_formulas()

    def calculate(self, inputs: Dict[str, Any]) -> CalculationResult:
        \"\"\"
        Perform calculation using verified formula.

        NEVER modifies or estimates values.
        Uses ONLY the provided formulas.
        \"\"\"
        # Validate inputs
        # Select appropriate formula
        # Apply formula exactly
        # Return with full provenance
        pass
"""
        }
        return templates.get(agent_type, templates[AgentType.DATA_PROCESSOR])


class TestGenerationPipeline:
    """
    Generate comprehensive tests for agents
    """

    async def generate_tests(self, spec: AgentSpecification, agent_code: str) -> str:
        """Generate test suite for agent"""

        test_prompt = f"""Generate comprehensive pytest tests for this GreenLang agent:

Agent Specification:
{json.dumps({
    'name': spec.name,
    'inputs': spec.inputs,
    'outputs': spec.outputs,
    'business_rules': spec.business_rules
}, indent=2)}

Agent Code:
```python
{agent_code}
```

Generate tests that cover:
1. Happy path scenarios
2. Edge cases
3. Error handling
4. Input validation
5. Output format validation
6. Performance requirements
7. Business rule compliance
8. Data provenance verification

Use pytest fixtures and parametrize for comprehensive coverage.
Include both unit tests and integration tests."""

        test_code = await self._generate_with_llm(test_prompt)
        return test_code


class DocumentationGenerator:
    """
    Generate comprehensive documentation
    """

    async def generate_docs(self, spec: AgentSpecification, code: str) -> str:
        """Generate documentation for agent"""

        doc_prompt = f"""Generate comprehensive documentation for this GreenLang agent:

Agent: {spec.name}
Description: {spec.description}

Code:
```python
{code}
```

Generate documentation including:
1. Overview and purpose
2. Architecture description
3. API documentation
4. Usage examples
5. Configuration guide
6. Performance characteristics
7. Error handling
8. Deployment instructions

Format as Markdown with clear sections."""

        documentation = await self._generate_with_llm(doc_prompt)
        return documentation


class QualityImprover:
    """
    Improve code quality using LLM suggestions
    """

    async def improve(self, code: str, spec: AgentSpecification) -> str:
        """Improve code quality"""

        improvement_prompt = f"""Improve this GreenLang agent code for production use:

Current Code:
```python
{code}
```

Requirements:
- Performance: {spec.performance_requirements}
- Compliance: {spec.compliance_requirements}

Improve the code by:
1. Optimizing performance
2. Adding comprehensive error handling
3. Improving logging and monitoring
4. Adding input validation
5. Ensuring thread safety
6. Adding retry logic where appropriate
7. Optimizing memory usage
8. Adding caching where beneficial

CRITICAL: Do not modify calculation formulas or add estimated values.

Return the improved code."""

        improved_code = await self._generate_with_llm(improvement_prompt)
        return improved_code


class AgentValidator:
    """
    Validate generated agents
    """

    async def validate(self, code: str, tests: str) -> Dict[str, Any]:
        """Validate agent code and tests"""

        validation_results = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }

        # Syntax validation
        if not self._validate_syntax(code):
            validation_results['passed'] = False
            validation_results['issues'].append('Syntax error in agent code')

        # Security validation
        security_issues = self._check_security(code)
        if security_issues:
            validation_results['passed'] = False
            validation_results['issues'].extend(security_issues)

        # Performance validation
        perf_warnings = self._check_performance(code)
        validation_results['warnings'].extend(perf_warnings)

        # Test coverage validation
        coverage = self._check_test_coverage(code, tests)
        validation_results['metrics']['test_coverage'] = coverage
        if coverage < 0.8:
            validation_results['warnings'].append(f'Low test coverage: {coverage:.1%}')

        # Compliance validation
        compliance = self._check_compliance(code)
        if not compliance['compliant']:
            validation_results['passed'] = False
            validation_results['issues'].extend(compliance['violations'])

        return validation_results

    def _check_security(self, code: str) -> List[str]:
        """Check for security issues"""
        issues = []

        # Check for dangerous functions
        dangerous_patterns = [
            (r'eval\(', 'Use of eval() is prohibited'),
            (r'exec\(', 'Use of exec() is prohibited'),
            (r'__import__', 'Dynamic imports are restricted'),
            (r'subprocess\.', 'Subprocess calls need review'),
            (r'os\.system', 'System calls are prohibited')
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(message)

        return issues

    def _check_compliance(self, code: str) -> Dict[str, Any]:
        """Check compliance requirements"""

        compliance_result = {
            'compliant': True,
            'violations': []
        }

        # Check for required logging
        if 'logger' not in code:
            compliance_result['violations'].append('Missing logging implementation')

        # Check for provenance tracking
        if 'provenance' not in code.lower():
            compliance_result['violations'].append('Missing provenance tracking')

        # Check for validation
        if 'validate' not in code.lower():
            compliance_result['violations'].append('Missing input validation')

        if compliance_result['violations']:
            compliance_result['compliant'] = False

        return compliance_result


class PromptVersionManager:
    """
    Manage and version prompts for reproducibility
    """

    def __init__(self):
        self.prompts = {}
        self.versions = {}
        self.performance_history = {}

    def register_prompt(self, name: str, prompt: str, metadata: Dict[str, Any]):
        """Register a new prompt version"""

        version = self._generate_version(name)

        self.prompts[f"{name}:{version}"] = {
            'prompt': prompt,
            'metadata': metadata,
            'created_at': datetime.utcnow(),
            'performance': {
                'accuracy': 0.0,
                'consistency': 0.0,
                'generation_quality': 0.0
            }
        }

        self.versions[name] = version
        return version

    def get_prompt(self, name: str, version: Optional[str] = None) -> str:
        """Get specific prompt version"""

        if version is None:
            version = self.versions.get(name)

        key = f"{name}:{version}"
        if key in self.prompts:
            return self.prompts[key]['prompt']

        raise ValueError(f"Prompt {name}:{version} not found")

    def update_performance(self, name: str, version: str, metrics: Dict[str, float]):
        """Update performance metrics for prompt"""

        key = f"{name}:{version}"
        if key in self.prompts:
            self.prompts[key]['performance'].update(metrics)

    def select_best_version(self, name: str) -> str:
        """Select best performing version of prompt"""

        versions = [k for k in self.prompts.keys() if k.startswith(f"{name}:")]

        best_version = None
        best_score = 0

        for version_key in versions:
            performance = self.prompts[version_key]['performance']
            score = (
                performance['accuracy'] * 0.5 +
                performance['consistency'] * 0.3 +
                performance['generation_quality'] * 0.2
            )

            if score > best_score:
                best_score = score
                best_version = version_key.split(':')[1]

        return best_version


# Example usage
async def example_agent_creation():
    """Example of creating an agent using the factory"""

    spec = AgentSpecification(
        name="Scope3Calculator",
        type=AgentType.CALCULATOR,
        description="Calculate Scope 3 emissions for purchased goods",
        inputs={
            'purchase_data': 'DataFrame with purchase records',
            'emission_factors': 'Dictionary of emission factors',
            'calculation_method': 'String specifying calculation approach'
        },
        outputs={
            'emissions': 'Calculated emissions in tCO2e',
            'breakdown': 'Emissions by category',
            'provenance': 'Data lineage and sources'
        },
        dependencies=['pandas', 'numpy'],
        business_rules=[
            'Use EPA emission factors for US operations',
            'Apply DEFRA factors for UK operations',
            'Prefer supplier-specific factors when available',
            'Flag any calculations with confidence < 80%'
        ],
        performance_requirements={
            'max_processing_time': 5,  # seconds
            'max_memory_usage': 500  # MB
        },
        compliance_requirements=[
            'GHG Protocol compliant',
            'CSRD ready',
            'Full audit trail required'
        ]
    )

    factory = AgentFactoryLLM()
    agent_bundle = await factory.create_agent(spec)

    return agent_bundle


# Export main components
__all__ = [
    'AgentFactoryLLM',
    'AgentSpecification',
    'AgentType',
    'CodeGenerationPipeline',
    'TestGenerationPipeline',
    'DocumentationGenerator',
    'PromptVersionManager'
]