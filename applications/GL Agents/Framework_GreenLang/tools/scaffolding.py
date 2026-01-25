"""
GreenLang Framework - Agent Scaffolding Generator

Generates complete agent directory structure from templates.
"""

from pathlib import Path
from typing import Dict, List, Optional, Type
import json
import shutil

from ..templates import BaseAgentTemplate, AgentConfig, CalculatorTemplate, OptimizerTemplate
from ..templates.base_agent import AgentType


class AgentScaffolder:
    """
    Generates complete agent scaffolding from templates.

    Usage:
        >>> config = AgentConfig(
        ...     agent_id="GL-007",
        ...     name="STEAMTRAP",
        ...     full_name="Steam Trap Monitoring and Analysis",
        ...     description="Detects and quantifies steam trap failures",
        ... )
        >>> scaffolder = AgentScaffolder(output_dir="c:/agents")
        >>> scaffolder.create_agent(config)
    """

    TEMPLATE_MAP: Dict[AgentType, Type[BaseAgentTemplate]] = {
        AgentType.CALCULATOR: CalculatorTemplate,
        AgentType.OPTIMIZER: OptimizerTemplate,
        AgentType.ANALYZER: BaseAgentTemplate,
        AgentType.PREDICTOR: BaseAgentTemplate,
        AgentType.ORCHESTRATOR: BaseAgentTemplate,
        AgentType.HYBRID: BaseAgentTemplate,
    }

    def __init__(self, output_dir: str):
        """
        Initialize scaffolder.

        Args:
            output_dir: Directory where agents will be created
        """
        self.output_dir = Path(output_dir)

    def create_agent(
        self,
        config: AgentConfig,
        overwrite: bool = False,
    ) -> Path:
        """
        Create complete agent from configuration.

        Args:
            config: Agent configuration
            overwrite: Whether to overwrite existing agent

        Returns:
            Path to created agent directory
        """
        agent_dir = self.output_dir / f"{config.agent_id}_{config.name}"

        if agent_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Agent already exists: {agent_dir}")
            shutil.rmtree(agent_dir)

        # Select template class
        template_class = self.TEMPLATE_MAP.get(config.agent_type, BaseAgentTemplate)
        template = template_class(config)

        # Create directory structure
        self._create_directories(agent_dir, template)

        # Generate files
        self._create_files(agent_dir, template)

        # Save configuration
        self._save_config(agent_dir, config)

        return agent_dir

    def _create_directories(
        self,
        agent_dir: Path,
        template: BaseAgentTemplate,
    ) -> None:
        """Create directory structure."""
        agent_dir.mkdir(parents=True, exist_ok=True)

        for dir_name in template.get_directory_structure():
            (agent_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def _create_files(
        self,
        agent_dir: Path,
        template: BaseAgentTemplate,
    ) -> None:
        """Create files from templates."""
        templates = template.get_all_templates()

        for file_path, content in templates.items():
            full_path = agent_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')

        # Create __init__.py files for all directories
        for dir_name in template.get_directory_structure():
            init_path = agent_dir / dir_name / "__init__.py"
            if not init_path.exists():
                init_path.write_text(f'"""{dir_name} module."""\n', encoding='utf-8')

    def _save_config(self, agent_dir: Path, config: AgentConfig) -> None:
        """Save agent configuration."""
        config_path = agent_dir / "agent_config.json"
        config_path.write_text(
            json.dumps(config.to_dict(), indent=2),
            encoding='utf-8'
        )

    def validate_config(self, config: AgentConfig) -> List[str]:
        """
        Validate agent configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check agent ID format
        if not config.agent_id.startswith("GL-"):
            errors.append("Agent ID must start with 'GL-'")

        if not config.agent_id[3:].isdigit():
            errors.append("Agent ID must be in format 'GL-NNN'")

        # Check name
        if not config.name.isupper():
            errors.append("Agent name should be uppercase")

        if not config.name.replace("_", "").isalpha():
            errors.append("Agent name should only contain letters and underscores")

        # Check version format
        parts = config.version.split(".")
        if len(parts) != 3:
            errors.append("Version must be in semver format (X.Y.Z)")

        # Check description
        if len(config.description) < 20:
            errors.append("Description should be at least 20 characters")

        return errors

    def list_available_templates(self) -> Dict[str, str]:
        """List available templates with descriptions."""
        return {
            "calculator": "Deterministic calculation agent with provenance tracking",
            "optimizer": "Multi-objective optimization with constraints",
            "analyzer": "Data analysis and pattern detection",
            "predictor": "Predictive modeling and forecasting",
            "orchestrator": "Multi-agent coordination and workflow",
            "hybrid": "Combined functionality from multiple types",
        }

    def get_template_structure(self, agent_type: AgentType) -> Dict[str, str]:
        """Get directory structure for a template type."""
        template_class = self.TEMPLATE_MAP.get(agent_type, BaseAgentTemplate)

        # Create temporary config
        temp_config = AgentConfig(
            agent_id="GL-000",
            name="TEMPLATE",
            full_name="Template Agent",
            description="Template for structure preview",
            agent_type=agent_type,
        )

        template = template_class(temp_config)
        return template.get_directory_structure()


class BatchScaffolder:
    """
    Create multiple agents from configuration file.
    """

    def __init__(self, output_dir: str):
        """Initialize batch scaffolder."""
        self.scaffolder = AgentScaffolder(output_dir)

    def create_from_manifest(
        self,
        manifest_path: str,
        overwrite: bool = False,
    ) -> List[Path]:
        """
        Create agents from manifest file.

        Args:
            manifest_path: Path to JSON manifest
            overwrite: Whether to overwrite existing agents

        Returns:
            List of created agent directories
        """
        with open(manifest_path) as f:
            manifest = json.load(f)

        created = []
        for agent_spec in manifest.get("agents", []):
            config = AgentConfig(
                agent_id=agent_spec["agent_id"],
                name=agent_spec["name"],
                full_name=agent_spec["full_name"],
                description=agent_spec["description"],
                agent_type=AgentType(agent_spec.get("type", "calculator")),
                standards=agent_spec.get("standards", []),
                regulations=agent_spec.get("regulations", []),
            )

            path = self.scaffolder.create_agent(config, overwrite=overwrite)
            created.append(path)

        return created
