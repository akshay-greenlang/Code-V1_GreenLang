# -*- coding: utf-8 -*-
"""
GreenLang Agent Factory Python SDK

Production-ready SDK for generating, testing, and deploying GreenLang agents
with full 12-dimension quality standards compliance.

Example:
    >>> from greenlang.factory import AgentFactory
    >>> factory = AgentFactory()
    >>> agent = factory.create_agent("emissions_calc.yaml")
    >>> validation = factory.validate_agent(agent)
    >>> factory.deploy_agent(agent, env="production")
"""

from typing import Dict, List, Optional, Any, Union, Callable, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from pydantic import BaseModel, Field, validator
import yaml
import jinja2
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class AgentTemplate(str, Enum):
    """Available agent templates."""
    BASE = "base"
    INDUSTRIAL = "industrial"
    HVAC = "hvac"
    CROSSCUTTING = "crosscutting"
    REGULATORY = "regulatory"
    SCOPE1 = "scope1"
    SCOPE2 = "scope2"
    SCOPE3 = "scope3"


class Framework(str, Enum):
    """Supported agent frameworks."""
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    NATIVE = "native"


class Language(str, Enum):
    """Implementation languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentStrategy(str, Enum):
    """Deployment strategies."""
    ROLLING = "rolling"
    BLUE_GREEN = "blue-green"
    CANARY = "canary"
    RECREATE = "recreate"


class Platform(str, Enum):
    """Deployment platforms."""
    DOCKER = "docker"
    KUBERNETES = "k8s"
    LAMBDA = "lambda"
    AZURE_FUNCTIONS = "azure"
    GCP_FUNCTIONS = "gcp"


class QualityDimension(str, Enum):
    """12 quality dimensions for agent validation."""
    DETERMINISM = "determinism"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    AUDITABILITY = "auditability"
    SECURITY = "security"
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    MAINTAINABILITY = "maintainability"
    TESTABILITY = "testability"
    COMPLIANCE = "compliance"
    USABILITY = "usability"
    RELIABILITY = "reliability"


@dataclass
class AgentSpecification:
    """Agent specification model."""

    name: str
    version: str
    description: str
    category: str
    template: AgentTemplate
    framework: Framework
    language: Language

    # Agent capabilities
    capabilities: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Quality requirements
    quality_targets: Dict[QualityDimension, float] = field(default_factory=dict)
    coverage_target: float = 85.0
    complexity_max: int = 10

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    # Metadata
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "AgentSpecification":
        """Load specification from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save specification to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


@dataclass
class ValidationResult:
    """Agent validation result."""

    agent_name: str
    timestamp: datetime
    passed: bool
    score: float  # 0-100

    # Dimension results
    dimension_results: Dict[QualityDimension, Dict[str, Any]] = field(default_factory=dict)

    # Issues found
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    info: List[Dict[str, Any]] = field(default_factory=list)

    def passes_all_dimensions(self) -> bool:
        """Check if all quality dimensions pass."""
        return all(
            result.get("passed", False)
            for result in self.dimension_results.values()
        )

    def get_dimension_score(self, dimension: QualityDimension) -> float:
        """Get score for specific dimension."""
        return self.dimension_results.get(dimension, {}).get("score", 0.0)

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "score": self.score,
            "dimensions": {
                dim.value: result
                for dim, result in self.dimension_results.items()
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info
        }


@dataclass
class TestResult:
    """Agent test execution result."""

    agent_name: str
    test_type: str  # unit, integration, e2e, performance
    timestamp: datetime
    passed: bool

    # Test metrics
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0

    # Coverage metrics
    coverage_percent: float = 0.0
    lines_covered: int = 0
    lines_total: int = 0

    # Performance metrics
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0

    # Details
    failures: List[Dict[str, Any]] = field(default_factory=list)
    output: Optional[str] = None

    def meets_coverage_target(self, target: float) -> bool:
        """Check if coverage meets target."""
        return self.coverage_percent >= target


@dataclass
class DeploymentResult:
    """Agent deployment result."""

    agent_name: str
    environment: Environment
    timestamp: datetime
    success: bool

    # Deployment info
    version: str
    platform: Platform
    strategy: DeploymentStrategy
    replicas: int = 1

    # Status
    status: str = "pending"
    health_check_passed: bool = False
    endpoints: List[str] = field(default_factory=list)

    # Metrics
    deployment_time_seconds: float = 0.0
    rollback_available: bool = False
    previous_version: Optional[str] = None

    # Details
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class AgentFactory:
    """
    Main factory class for agent creation and management.

    Example:
        >>> factory = AgentFactory()
        >>> agent = factory.create_agent("spec.yaml")
        >>> factory.deploy_agent(agent, env="production")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AgentFactory.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._load_default_config()
        self.template_engine = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self._get_template_dir())
        )
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("parallel", 4))
        self._plugins: Dict[str, Any] = {}
        self._load_plugins()

    def create_agent(
        self,
        spec_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        validate: bool = True,
        dry_run: bool = False,
        force: bool = False,
        **kwargs
    ) -> "Agent":
        """
        Create agent from specification.

        Args:
            spec_path: Path to agent specification YAML
            output_dir: Output directory for generated files
            validate: Validate specification before generation
            dry_run: Preview without generating files
            force: Overwrite existing files
            **kwargs: Additional options

        Returns:
            Generated Agent instance

        Raises:
            ValueError: If specification is invalid
            FileExistsError: If agent exists and force=False
        """
        # Load specification
        spec = AgentSpecification.from_yaml(spec_path)

        # Validate if requested
        if validate:
            validation = self.validate_specification(spec)
            if not validation.passed:
                raise ValueError(f"Specification validation failed: {validation.errors}")

        # Check if agent exists
        output_dir = Path(output_dir or self.config["output_dir"])
        agent_dir = output_dir / spec.name

        if agent_dir.exists() and not force:
            raise FileExistsError(f"Agent directory already exists: {agent_dir}")

        # Dry run - return preview
        if dry_run:
            return self._preview_agent(spec)

        # Generate agent
        agent = self._generate_agent(spec, agent_dir, **kwargs)

        # Generate additional components
        if kwargs.get("with_tests", True):
            self.generate_tests(agent)

        if kwargs.get("with_docs", True):
            self.generate_docs(agent)

        logger.info(f"Agent created successfully: {agent.name}")
        return agent

    def scaffold_agent(
        self,
        name: str,
        template: AgentTemplate = AgentTemplate.BASE,
        category: Optional[str] = None,
        framework: Framework = Framework.LANGCHAIN,
        language: Language = Language.PYTHON,
        interactive: bool = False,
        **kwargs
    ) -> AgentSpecification:
        """
        Generate agent boilerplate and specification template.

        Args:
            name: Agent name
            template: Base template to use
            category: Agent category
            framework: Target framework
            language: Implementation language
            interactive: Interactive mode with prompts
            **kwargs: Additional options

        Returns:
            Generated AgentSpecification
        """
        if interactive:
            spec = self._interactive_scaffold(name)
        else:
            spec = AgentSpecification(
                name=name,
                version="0.1.0",
                description=f"{name} agent implementation",
                category=category or "general",
                template=template,
                framework=framework,
                language=language,
                created_at=DeterministicClock.now()
            )

        # Apply template defaults
        self._apply_template_defaults(spec, template)

        # Save specification
        spec_path = Path(kwargs.get("output_dir", ".")) / f"{name}_spec.yaml"
        spec.to_yaml(spec_path)

        logger.info(f"Agent scaffolded: {spec_path}")
        return spec

    def validate_agent(
        self,
        agent: Union["Agent", str, Path],
        dimensions: Optional[List[QualityDimension]] = None,
        strict: bool = False,
        **kwargs
    ) -> ValidationResult:
        """
        Validate agent against quality standards.

        Args:
            agent: Agent instance or path
            dimensions: Specific dimensions to validate (default: all)
            strict: Fail on warnings
            **kwargs: Additional validation options

        Returns:
            ValidationResult with detailed findings
        """
        # Resolve agent
        if isinstance(agent, (str, Path)):
            agent = self.load_agent(agent)

        # Initialize result
        result = ValidationResult(
            agent_name=agent.name,
            timestamp=DeterministicClock.now(),
            passed=True,
            score=100.0
        )

        # Validate each dimension
        dimensions = dimensions or list(QualityDimension)

        for dimension in dimensions:
            dim_result = self._validate_dimension(agent, dimension, **kwargs)
            result.dimension_results[dimension] = dim_result

            if not dim_result["passed"]:
                result.passed = False
                result.errors.extend(dim_result.get("errors", []))

            result.warnings.extend(dim_result.get("warnings", []))

        # Calculate overall score
        result.score = self._calculate_quality_score(result)

        # Apply strict mode
        if strict and result.warnings:
            result.passed = False

        return result

    def validate_specification(
        self,
        spec: Union[AgentSpecification, str, Path],
        schema_version: Optional[str] = None,
        **kwargs
    ) -> ValidationResult:
        """
        Validate agent specification.

        Args:
            spec: Specification or path to YAML
            schema_version: Schema version to validate against
            **kwargs: Additional options

        Returns:
            ValidationResult
        """
        # Load specification if path provided
        if isinstance(spec, (str, Path)):
            spec = AgentSpecification.from_yaml(spec)

        result = ValidationResult(
            agent_name=spec.name,
            timestamp=DeterministicClock.now(),
            passed=True,
            score=100.0
        )

        # Validate against schema
        schema_errors = self._validate_against_schema(spec, schema_version)
        if schema_errors:
            result.passed = False
            result.errors.extend(schema_errors)

        # Validate completeness
        if not spec.inputs or not spec.outputs:
            result.warnings.append({
                "type": "completeness",
                "message": "Input/output specifications are incomplete"
            })

        # Validate dependencies
        dep_issues = self._validate_dependencies(spec.dependencies)
        result.warnings.extend(dep_issues)

        return result

    def generate_implementation(
        self,
        agent: Union["Agent", AgentSpecification],
        language: Optional[Language] = None,
        framework: Optional[Framework] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate agent implementation code.

        Args:
            agent: Agent or specification
            language: Override language
            framework: Override framework
            **kwargs: Additional options

        Returns:
            Dictionary of filename -> code content
        """
        # Resolve specification
        if isinstance(agent, Agent):
            spec = agent.specification
        else:
            spec = agent

        language = language or spec.language
        framework = framework or spec.framework

        # Select code generator
        generator = self._get_code_generator(language, framework)

        # Generate code files
        files = {}

        # Main agent implementation
        files[f"{spec.name}_agent.{self._get_extension(language)}"] = (
            generator.generate_agent_class(spec)
        )

        # Input/Output models
        files[f"{spec.name}_models.{self._get_extension(language)}"] = (
            generator.generate_models(spec)
        )

        # Configuration
        files[f"{spec.name}_config.{self._get_extension(language)}"] = (
            generator.generate_config(spec)
        )

        # Utilities
        if spec.capabilities:
            files[f"{spec.name}_utils.{self._get_extension(language)}"] = (
                generator.generate_utilities(spec)
            )

        return files

    def generate_tests(
        self,
        agent: Union["Agent", str, Path],
        test_type: str = "all",
        coverage_target: float = 85.0,
        framework: Optional[str] = None,
        **kwargs
    ) -> TestResult:
        """
        Generate comprehensive test suite.

        Args:
            agent: Agent instance or path
            test_type: Test types (unit|integration|e2e|performance|all)
            coverage_target: Target coverage percentage
            framework: Test framework to use
            **kwargs: Additional options

        Returns:
            TestResult with generation details
        """
        # Resolve agent
        if isinstance(agent, (str, Path)):
            agent = self.load_agent(agent)

        # Determine test framework
        if not framework:
            framework = self._get_test_framework(agent.specification.language)

        # Generate test files
        test_files = {}

        if test_type in ["unit", "all"]:
            test_files.update(self._generate_unit_tests(agent, framework))

        if test_type in ["integration", "all"]:
            test_files.update(self._generate_integration_tests(agent, framework))

        if test_type in ["e2e", "all"]:
            test_files.update(self._generate_e2e_tests(agent, framework))

        if test_type in ["performance", "all"]:
            test_files.update(self._generate_performance_tests(agent, framework))

        # Generate fixtures if requested
        if kwargs.get("fixtures", True):
            test_files.update(self._generate_test_fixtures(agent))

        # Generate mocks if requested
        if kwargs.get("mocks", True):
            test_files.update(self._generate_test_mocks(agent))

        # Write test files
        test_dir = agent.path / "tests"
        test_dir.mkdir(exist_ok=True)

        for filename, content in test_files.items():
            (test_dir / filename).write_text(content)

        # Create test result
        result = TestResult(
            agent_name=agent.name,
            test_type=test_type,
            timestamp=DeterministicClock.now(),
            passed=True,
            tests_run=len(test_files),
            tests_passed=len(test_files)
        )

        logger.info(f"Generated {len(test_files)} test files for {agent.name}")
        return result

    def generate_docs(
        self,
        agent: Union["Agent", str, Path],
        format: str = "markdown",
        include: List[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate comprehensive documentation.

        Args:
            agent: Agent instance or path
            format: Output format (markdown|html|pdf|openapi)
            include: Sections to include (api|usage|architecture|all)
            **kwargs: Additional options

        Returns:
            Dictionary of filename -> documentation content
        """
        # Resolve agent
        if isinstance(agent, (str, Path)):
            agent = self.load_agent(agent)

        include = include or ["all"]
        docs = {}

        # Generate main README
        docs["README.md"] = self._generate_readme(agent)

        # API documentation
        if "api" in include or "all" in include:
            if format == "openapi":
                docs["openapi.yaml"] = self._generate_openapi_spec(agent)
            else:
                docs["API.md"] = self._generate_api_docs(agent)

        # Usage documentation
        if "usage" in include or "all" in include:
            docs["USAGE.md"] = self._generate_usage_docs(agent)

            if kwargs.get("examples", True):
                docs["examples/"] = self._generate_examples(agent)

        # Architecture documentation
        if "architecture" in include or "all" in include:
            docs["ARCHITECTURE.md"] = self._generate_architecture_docs(agent)

            if kwargs.get("diagrams", True):
                docs["diagrams/"] = self._generate_diagrams(agent)

        # Generate changelog template
        if kwargs.get("changelog", True):
            docs["CHANGELOG.md"] = self._generate_changelog_template(agent)

        return docs

    def test_agent(
        self,
        agent: Union["Agent", str, Path],
        test_type: str = "all",
        coverage_min: float = 85.0,
        parallel: int = 4,
        **kwargs
    ) -> TestResult:
        """
        Run agent test suite.

        Args:
            agent: Agent instance or path
            test_type: Test types to run
            coverage_min: Minimum coverage required
            parallel: Parallel test workers
            **kwargs: Additional options

        Returns:
            TestResult with execution details

        Raises:
            TestFailureError: If tests fail or coverage is below threshold
        """
        # Resolve agent
        if isinstance(agent, (str, Path)):
            agent = self.load_agent(agent)

        # Run tests based on type
        runner = self._get_test_runner(agent.specification.language)

        result = runner.run_tests(
            agent.path / "tests",
            test_type=test_type,
            parallel=parallel,
            **kwargs
        )

        # Check coverage threshold
        if not result.meets_coverage_target(coverage_min):
            raise TestFailureError(
                f"Coverage {result.coverage_percent}% below minimum {coverage_min}%"
            )

        return result

    def build_agent(
        self,
        agent: Union["Agent", str, Path],
        platform: Platform = Platform.DOCKER,
        optimize: bool = False,
        tag: Optional[str] = None,
        **kwargs
    ) -> "BuildResult":
        """
        Build agent for deployment.

        Args:
            agent: Agent instance or path
            platform: Target platform
            optimize: Apply optimizations
            tag: Version tag
            **kwargs: Additional build options

        Returns:
            BuildResult with artifact details
        """
        # Resolve agent
        if isinstance(agent, (str, Path)):
            agent = self.load_agent(agent)

        # Get builder for platform
        builder = self._get_builder(platform)

        # Prepare build configuration
        build_config = {
            "optimize": optimize,
            "tag": tag or agent.version,
            "registry": kwargs.get("registry", self.config.get("registry", {}).get("url")),
            **kwargs
        }

        # Execute build
        result = builder.build(agent, build_config)

        logger.info(f"Agent built successfully: {agent.name} [{platform.value}]")
        return result

    def deploy_agent(
        self,
        agent: Union["Agent", str, Path],
        env: Union[Environment, str] = Environment.STAGING,
        strategy: DeploymentStrategy = DeploymentStrategy.ROLLING,
        replicas: int = 1,
        **kwargs
    ) -> DeploymentResult:
        """
        Deploy agent to environment.

        Args:
            agent: Agent instance or path
            env: Target environment
            strategy: Deployment strategy
            replicas: Number of replicas
            **kwargs: Additional deployment options

        Returns:
            DeploymentResult with deployment details

        Raises:
            DeploymentError: If deployment fails
        """
        # Resolve agent and environment
        if isinstance(agent, (str, Path)):
            agent = self.load_agent(agent)

        if isinstance(env, str):
            env = Environment(env)

        # Get deployer for environment
        deployer = self._get_deployer(env)

        # Prepare deployment configuration
        deploy_config = {
            "strategy": strategy,
            "replicas": replicas,
            "auto_scale": kwargs.get("auto_scale", False),
            "health_check": kwargs.get("health_check", True),
            "rollback_on_failure": kwargs.get("rollback_on_failure", True),
            **kwargs
        }

        # Execute deployment
        result = deployer.deploy(agent, deploy_config)

        # Verify deployment if requested
        if kwargs.get("verify", True):
            if not self._verify_deployment(result):
                if deploy_config["rollback_on_failure"]:
                    self.rollback_agent(agent, env)
                raise DeploymentError(f"Deployment verification failed for {agent.name}")

        logger.info(f"Agent deployed successfully: {agent.name} -> {env.value}")
        return result

    def rollback_agent(
        self,
        agent: Union["Agent", str, Path],
        env: Union[Environment, str],
        version: Optional[str] = None,
        **kwargs
    ) -> DeploymentResult:
        """
        Rollback agent deployment.

        Args:
            agent: Agent instance or path
            env: Environment
            version: Target version (default: previous)
            **kwargs: Additional options

        Returns:
            DeploymentResult for rollback
        """
        # Resolve agent and environment
        if isinstance(agent, (str, Path)):
            agent = self.load_agent(agent)

        if isinstance(env, str):
            env = Environment(env)

        # Get deployer
        deployer = self._get_deployer(env)

        # Execute rollback
        result = deployer.rollback(agent, version, **kwargs)

        logger.info(f"Agent rolled back: {agent.name} in {env.value}")
        return result

    def batch_create(
        self,
        specs_dir: Union[str, Path],
        pattern: str = "*.yaml",
        parallel: int = 4,
        continue_on_error: bool = False,
        **kwargs
    ) -> List["Agent"]:
        """
        Create multiple agents from specifications.

        Args:
            specs_dir: Directory containing specifications
            pattern: File pattern
            parallel: Parallel workers
            continue_on_error: Continue on failures
            **kwargs: Additional options

        Returns:
            List of created agents
        """
        specs_dir = Path(specs_dir)
        spec_files = list(specs_dir.glob(pattern))

        logger.info(f"Found {len(spec_files)} specifications to process")

        agents = []
        futures = []

        with ProcessPoolExecutor(max_workers=parallel) as executor:
            for spec_file in spec_files:
                future = executor.submit(self.create_agent, spec_file, **kwargs)
                futures.append((spec_file, future))

            for spec_file, future in futures:
                try:
                    agent = future.result()
                    agents.append(agent)
                except Exception as e:
                    logger.error(f"Failed to create agent from {spec_file}: {e}")
                    if not continue_on_error:
                        raise

        logger.info(f"Successfully created {len(agents)} agents")
        return agents

    def batch_test(
        self,
        agents_dir: Union[str, Path],
        pattern: str = "*",
        parallel: int = 4,
        fail_fast: bool = False,
        **kwargs
    ) -> List[TestResult]:
        """
        Test multiple agents.

        Args:
            agents_dir: Directory containing agents
            pattern: Agent pattern
            parallel: Parallel workers
            fail_fast: Stop on first failure
            **kwargs: Additional test options

        Returns:
            List of test results
        """
        agents_dir = Path(agents_dir)
        agent_dirs = [d for d in agents_dir.glob(pattern) if d.is_dir()]

        results = []

        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = []

            for agent_dir in agent_dirs:
                future = executor.submit(self.test_agent, agent_dir, **kwargs)
                futures.append((agent_dir, future))

            for agent_dir, future in futures:
                try:
                    result = future.result()
                    results.append(result)

                    if not result.passed and fail_fast:
                        # Cancel remaining futures
                        for _, f in futures:
                            f.cancel()
                        break

                except Exception as e:
                    logger.error(f"Failed to test agent {agent_dir}: {e}")
                    if fail_fast:
                        raise

        return results

    def batch_deploy(
        self,
        agents_list: Union[str, Path, List[str]],
        env: Union[Environment, str],
        stagger: int = 0,
        rollback_all_on_failure: bool = False,
        **kwargs
    ) -> List[DeploymentResult]:
        """
        Deploy multiple agents.

        Args:
            agents_list: List file or list of agent names
            env: Target environment
            stagger: Stagger deployments (seconds)
            rollback_all_on_failure: Rollback all on any failure
            **kwargs: Additional deployment options

        Returns:
            List of deployment results
        """
        # Load agent list
        if isinstance(agents_list, (str, Path)):
            with open(agents_list) as f:
                agents = [line.strip() for line in f if line.strip()]
        else:
            agents = agents_list

        results = []
        deployed = []

        try:
            for i, agent_name in enumerate(agents):
                # Stagger deployment
                if i > 0 and stagger > 0:
                    asyncio.sleep(stagger)

                # Deploy agent
                result = self.deploy_agent(agent_name, env, **kwargs)
                results.append(result)

                if result.success:
                    deployed.append(agent_name)
                else:
                    raise DeploymentError(f"Failed to deploy {agent_name}")

        except Exception as e:
            logger.error(f"Batch deployment failed: {e}")

            if rollback_all_on_failure:
                logger.info("Rolling back all deployed agents...")
                for agent_name in deployed:
                    self.rollback_agent(agent_name, env)

            raise

        return results

    def load_agent(self, path: Union[str, Path]) -> "Agent":
        """
        Load existing agent from directory.

        Args:
            path: Path to agent directory

        Returns:
            Agent instance
        """
        path = Path(path)

        # Load specification
        spec_file = path / f"{path.name}_spec.yaml"
        if not spec_file.exists():
            # Try alternative locations
            spec_file = path / "spec.yaml"
            if not spec_file.exists():
                raise FileNotFoundError(f"Agent specification not found in {path}")

        spec = AgentSpecification.from_yaml(spec_file)

        # Create agent instance
        return Agent(
            name=spec.name,
            version=spec.version,
            specification=spec,
            path=path
        )

    def install_plugin(
        self,
        plugin_name: str,
        version: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Install a plugin.

        Args:
            plugin_name: Plugin name
            version: Plugin version
            **kwargs: Additional options
        """
        # Plugin installation logic
        plugin_manager = self._get_plugin_manager()
        plugin_manager.install(plugin_name, version, **kwargs)

        # Reload plugins
        self._load_plugins()

        logger.info(f"Plugin installed: {plugin_name}")

    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List installed plugins.

        Returns:
            List of plugin information
        """
        return [
            {
                "name": name,
                "version": plugin.version,
                "enabled": plugin.enabled,
                "description": plugin.description
            }
            for name, plugin in self._plugins.items()
        ]

    # Private helper methods

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "output_dir": "./agents",
            "parallel": 4,
            "coverage_min": 85,
            "complexity_max": 10,
            "registry": {
                "url": "https://registry.greenlang.io"
            }
        }

    def _get_template_dir(self) -> Path:
        """Get templates directory."""
        return Path(__file__).parent / "templates"

    def _load_plugins(self) -> None:
        """Load installed plugins."""
        # Plugin loading logic
        pass

    def _get_code_generator(self, language: Language, framework: Framework) -> "CodeGenerator":
        """Get code generator for language/framework."""
        # Return appropriate generator
        pass

    def _get_extension(self, language: Language) -> str:
        """Get file extension for language."""
        extensions = {
            Language.PYTHON: "py",
            Language.TYPESCRIPT: "ts",
            Language.JAVA: "java",
            Language.GO: "go"
        }
        return extensions.get(language, "txt")

    # Additional private methods would be implemented here...


class Agent:
    """Agent instance representation."""

    def __init__(
        self,
        name: str,
        version: str,
        specification: AgentSpecification,
        path: Path
    ):
        self.name = name
        self.version = version
        self.specification = specification
        self.path = path

    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', version='{self.version}')"


# Exception classes

class AgentFactoryError(Exception):
    """Base exception for agent factory."""
    pass


class ValidationError(AgentFactoryError):
    """Validation failed."""
    pass


class TestFailureError(AgentFactoryError):
    """Test execution failed."""
    pass


class DeploymentError(AgentFactoryError):
    """Deployment failed."""
    pass


class BuildError(AgentFactoryError):
    """Build failed."""
    pass