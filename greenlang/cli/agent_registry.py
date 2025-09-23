"""
Agent Registry for dynamic agent discovery and plugin support
"""

import importlib
import inspect
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pkg_resources
import yaml
import json


class AgentRegistry:
    """Registry for discovering and managing agents including plugins"""

    def __init__(self, custom_paths: Optional[List[str]] = None):
        """Initialize agent registry

        Args:
            custom_paths: Additional paths to search for custom agents
        """
        self.agents = {}
        self.custom_paths = custom_paths or []
        self._discovered = False

        # Add default custom paths
        self.custom_paths.extend(
            [
                "agents/custom",
                os.path.expanduser("~/.greenlang/agents"),
                os.environ.get("GREENLANG_AGENTS_PATH", ""),
            ]
        )

        # Filter empty paths
        self.custom_paths = [p for p in self.custom_paths if p]

    def discover_agents(self) -> List[Dict[str, Any]]:
        """Discover all available agents from various sources

        Returns:
            List of agent information dictionaries
        """
        if not self._discovered:
            self._discover_core_agents()
            self._discover_entry_point_agents()
            self._discover_custom_agents()
            self._discovered = True

        return list(self.agents.values())

    def _discover_core_agents(self):
        """Discover core agents from greenlang.agents module"""
        try:
            from greenlang import agents

            # Get all agent classes
            for name, obj in inspect.getmembers(agents):
                if (
                    inspect.isclass(obj)
                    and name.endswith("Agent")
                    and name != "BaseAgent"
                ):

                    agent_id = name.replace("Agent", "").lower()

                    # Extract metadata
                    self.agents[agent_id] = {
                        "id": agent_id,
                        "name": name,
                        "module": f"greenlang.agents.{name}",
                        "class": obj,
                        "version": getattr(obj, "version", "0.0.1"),
                        "description": obj.__doc__ or "No description",
                        "type": "core",
                        "is_plugin": False,
                    }
        except ImportError:
            pass

    def _discover_entry_point_agents(self):
        """Discover agents registered via setuptools entry_points"""
        # Look for entry points in the 'greenlang.agents' group
        for entry_point in pkg_resources.iter_entry_points("greenlang.agents"):
            try:
                agent_class = entry_point.load()
                agent_id = entry_point.name

                self.agents[agent_id] = {
                    "id": agent_id,
                    "name": agent_class.__name__,
                    "module": entry_point.module_name,
                    "class": agent_class,
                    "version": getattr(agent_class, "version", "0.0.1"),
                    "description": agent_class.__doc__ or "Plugin agent",
                    "type": "plugin",
                    "is_plugin": True,
                    "entry_point": str(entry_point),
                }
            except Exception as e:
                # Log error but continue discovery
                print(f"Failed to load entry point {entry_point.name}: {e}")

    def _discover_custom_agents(self):
        """Discover custom agents from file system paths"""
        for custom_path in self.custom_paths:
            path = Path(custom_path)
            if not path.exists():
                continue

            # Look for Python files
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                self._load_custom_agent_file(py_file)

            # Look for YAML agent definitions
            for yaml_file in path.glob("*.yaml"):
                self._load_yaml_agent_definition(yaml_file)

    def _load_custom_agent_file(self, file_path: Path):
        """Load a custom agent from a Python file"""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find agent classes
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and name.endswith("Agent")
                    and hasattr(obj, "execute")
                ):

                    agent_id = f"custom_{file_path.stem}"

                    self.agents[agent_id] = {
                        "id": agent_id,
                        "name": name,
                        "module": str(file_path),
                        "class": obj,
                        "version": getattr(obj, "version", "0.0.1"),
                        "description": obj.__doc__ or "Custom agent",
                        "type": "custom",
                        "is_plugin": True,
                        "file_path": str(file_path),
                    }
        except Exception as e:
            print(f"Failed to load custom agent from {file_path}: {e}")

    def _load_yaml_agent_definition(self, file_path: Path):
        """Load agent definition from YAML file"""
        try:
            with open(file_path, "r") as f:
                definition = yaml.safe_load(f)

            if definition and "agent" in definition:
                agent_def = definition["agent"]
                agent_id = agent_def.get("id", file_path.stem)

                self.agents[agent_id] = {
                    "id": agent_id,
                    "name": agent_def.get("name", agent_id),
                    "version": agent_def.get("version", "0.0.1"),
                    "description": agent_def.get("description", "YAML-defined agent"),
                    "type": "yaml",
                    "is_plugin": True,
                    "definition": agent_def,
                    "file_path": str(file_path),
                }
        except Exception as e:
            print(f"Failed to load YAML agent from {file_path}: {e}")

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent

        Args:
            agent_id: Agent identifier

        Returns:
            Agent information dictionary or None if not found
        """
        if not self._discovered:
            self.discover_agents()

        return self.agents.get(agent_id)

    def get_agent_class(self, agent_id: str):
        """Get the agent class for instantiation

        Args:
            agent_id: Agent identifier

        Returns:
            Agent class or None if not found
        """
        agent_info = self.get_agent_info(agent_id)
        if agent_info:
            return agent_info.get("class")
        return None

    def instantiate_agent(self, agent_id: str, config: Optional[Dict] = None):
        """Create an instance of an agent

        Args:
            agent_id: Agent identifier
            config: Optional configuration for the agent

        Returns:
            Agent instance or None if not found
        """
        agent_class = self.get_agent_class(agent_id)
        if agent_class:
            try:
                if config:
                    return agent_class(config)
                else:
                    return agent_class()
            except Exception as e:
                print(f"Failed to instantiate agent {agent_id}: {e}")
        return None

    def get_agent_template(self, base_agent: str = "base") -> str:
        """Generate a template for creating a new agent

        Args:
            base_agent: Base agent to use as template

        Returns:
            Template code as string
        """
        template = '''"""
Custom GreenLang Agent Template
"""

from typing import Dict, Any
from greenlang.agents.base import BaseAgent, AgentResult


class CustomAgent(BaseAgent):
    """Custom agent for [describe purpose]
    
    This agent [describe what it does]
    """
    
    version = "0.0.1"
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the agent
        
        Args:
            config: Agent configuration
        """
        super().__init__(config or {
            "name": "CustomAgent",
            "description": "Custom agent implementation"
        })
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Add validation logic
        required_fields = ["field1", "field2"]
        return all(field in input_data for field in required_fields)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute the agent logic
        
        Args:
            input_data: Input data for processing
            
        Returns:
            AgentResult with output data
        """
        try:
            # Validate input
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Invalid input data"
                )
            
            # Process data
            result = self.process(input_data)
            
            return AgentResult(
                success=True,
                data=result
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e)
            )
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data
        
        Args:
            input_data: Input data
            
        Returns:
            Processed results
        """
        # Implement your logic here
        output = {
            "processed": True,
            "input_received": input_data,
            # Add your results
        }
        
        return output


# Optional: Register as entry point in setup.py
# entry_points={
#     'greenlang.agents': [
#         'custom = mypackage.custom_agent:CustomAgent',
#     ],
# }
'''
        return template

    def list_agent_ids(self) -> List[str]:
        """Get list of all agent IDs

        Returns:
            List of agent identifiers
        """
        if not self._discovered:
            self.discover_agents()

        return list(self.agents.keys())

    def export_registry(self, output_path: Path):
        """Export registry to JSON file

        Args:
            output_path: Path to save registry
        """
        if not self._discovered:
            self.discover_agents()

        # Prepare exportable data (exclude class objects)
        export_data = {}
        for agent_id, info in self.agents.items():
            export_info = {
                k: v for k, v in info.items() if k not in ["class", "definition"]
            }
            export_data[agent_id] = export_info

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)
