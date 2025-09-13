#!/usr/bin/env python3
"""
GreenLang SDK Pack Loading Example

This example demonstrates how to:
- Load and manage GreenLang packs programmatically
- Inspect pack contents and metadata
- Execute pack pipelines through the SDK
- Handle pack dependencies and validation
- Work with pack registries and repositories

This shows advanced SDK patterns for pack management and execution.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# GreenLang SDK imports
import greenlang
from greenlang import Orchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PackMetadata:
    """Pack metadata information"""
    name: str
    version: str
    kind: str
    license: str
    description: str
    author: str
    tags: List[str]
    compatibility: Dict[str, str]
    dependencies: List[str]
    pipelines: List[str]
    agents: List[str]


class PackLoader:
    """Enhanced pack loader with validation and dependency management"""

    def __init__(self):
        self.loaded_packs: Dict[str, PackMetadata] = {}
        self.pack_registry: Dict[str, str] = {}  # name -> path mapping
        self.orchestrator = Orchestrator()

    def register_pack_location(self, pack_name: str, pack_path: str):
        """Register a pack location for later loading"""
        self.pack_registry[pack_name] = pack_path
        logger.info(f"Registered pack location: {pack_name} -> {pack_path}")

    def load_pack_manifest(self, pack_path: str) -> PackMetadata:
        """Load and parse a pack manifest file"""
        manifest_path = Path(pack_path) / "pack.yaml"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Pack manifest not found: {manifest_path}")

        logger.info(f"Loading pack manifest from {manifest_path}")

        with open(manifest_path, 'r') as f:
            manifest_data = yaml.safe_load(f)

        # Extract metadata
        metadata = PackMetadata(
            name=manifest_data["name"],
            version=manifest_data["version"],
            kind=manifest_data.get("kind", "pack"),
            license=manifest_data["license"],
            description=manifest_data.get("description", ""),
            author=manifest_data.get("metadata", {}).get("author", "Unknown"),
            tags=manifest_data.get("metadata", {}).get("tags", []),
            compatibility=manifest_data.get("compat", {}),
            dependencies=manifest_data.get("dependencies", []),
            pipelines=manifest_data.get("contents", {}).get("pipelines", []),
            agents=manifest_data.get("contents", {}).get("agents", [])
        )

        return metadata

    def validate_pack_compatibility(self, metadata: PackMetadata) -> bool:
        """Validate pack compatibility with current environment"""
        logger.info(f"Validating compatibility for pack: {metadata.name}")

        # Check GreenLang version compatibility
        greenlang_compat = metadata.compatibility.get("greenlang")
        if greenlang_compat:
            current_version = greenlang.__version__
            logger.info(f"GreenLang compatibility: requires {greenlang_compat}, current {current_version}")
            # Note: In a real implementation, you'd use version parsing logic here

        # Check Python version compatibility
        python_compat = metadata.compatibility.get("python")
        if python_compat:
            import sys
            current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
            logger.info(f"Python compatibility: requires {python_compat}, current {current_python}")

        # For this example, we'll assume compatibility is valid
        return True

    def resolve_dependencies(self, metadata: PackMetadata) -> List[str]:
        """Resolve pack dependencies"""
        logger.info(f"Resolving dependencies for pack: {metadata.name}")

        resolved_deps = []
        for dep in metadata.dependencies:
            if isinstance(dep, str):
                # Simple string dependency
                resolved_deps.append(dep)
                logger.info(f"  Dependency: {dep}")
            elif isinstance(dep, dict):
                # Complex dependency with version constraints
                dep_name = dep.get("name")
                dep_version = dep.get("version", "*")
                resolved_deps.append(f"{dep_name}>={dep_version}")
                logger.info(f"  Dependency: {dep_name} (version: {dep_version})")

        return resolved_deps

    def load_pack(self, pack_path: str, validate: bool = True) -> PackMetadata:
        """Load a complete pack with validation and dependency resolution"""
        logger.info(f"Loading pack from: {pack_path}")

        # Load manifest
        metadata = self.load_pack_manifest(pack_path)

        # Validate compatibility if requested
        if validate:
            if not self.validate_pack_compatibility(metadata):
                raise ValueError(f"Pack {metadata.name} is not compatible with current environment")

        # Resolve dependencies
        resolved_deps = self.resolve_dependencies(metadata)

        # Load pipeline files
        pack_dir = Path(pack_path)
        available_pipelines = []

        for pipeline_file in metadata.pipelines:
            pipeline_path = pack_dir / pipeline_file
            if pipeline_path.exists():
                available_pipelines.append(str(pipeline_path))
                logger.info(f"  Found pipeline: {pipeline_file}")
            else:
                logger.warning(f"  Pipeline file not found: {pipeline_file}")

        # Store in loaded packs
        self.loaded_packs[metadata.name] = metadata

        logger.info(f"Successfully loaded pack: {metadata.name} v{metadata.version}")
        return metadata

    def list_loaded_packs(self) -> List[PackMetadata]:
        """List all loaded packs"""
        return list(self.loaded_packs.values())

    def get_pack_info(self, pack_name: str) -> Optional[PackMetadata]:
        """Get information about a specific loaded pack"""
        return self.loaded_packs.get(pack_name)

    def execute_pack_pipeline(
        self,
        pack_name: str,
        pipeline_name: str,
        inputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a pipeline from a loaded pack"""
        if pack_name not in self.loaded_packs:
            raise ValueError(f"Pack {pack_name} is not loaded")

        metadata = self.loaded_packs[pack_name]
        pack_path = self.pack_registry.get(pack_name)

        if not pack_path:
            raise ValueError(f"Pack path not found for {pack_name}")

        # Find the pipeline file
        pipeline_path = None
        for pipeline_file in metadata.pipelines:
            if pipeline_name in pipeline_file or pipeline_file == pipeline_name:
                pipeline_path = Path(pack_path) / pipeline_file
                break

        if not pipeline_path or not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline {pipeline_name} not found in pack {pack_name}")

        logger.info(f"Executing pipeline {pipeline_name} from pack {pack_name}")

        # Load and execute pipeline
        # Note: This is a simplified example - real implementation would use proper pipeline execution
        with open(pipeline_path, 'r') as f:
            pipeline_data = yaml.safe_load(f)

        # Simulate pipeline execution
        execution_result = {
            "pipeline_name": pipeline_data.get("name", pipeline_name),
            "pack_name": pack_name,
            "execution_timestamp": datetime.now().isoformat(),
            "status": "completed",
            "inputs_received": inputs,
            "outputs": {
                "simulation_result": "This is a simulated execution result",
                "pack_version": metadata.version,
                "pipeline_version": pipeline_data.get("version", 1)
            }
        }

        logger.info(f"Pipeline execution completed: {pipeline_name}")
        return execution_result


class PackRegistry:
    """Registry for managing multiple packs and repositories"""

    def __init__(self):
        self.repositories: Dict[str, str] = {}
        self.pack_loader = PackLoader()
        self.installed_packs: Dict[str, Dict[str, Any]] = {}

    def add_repository(self, name: str, url_or_path: str):
        """Add a pack repository"""
        self.repositories[name] = url_or_path
        logger.info(f"Added repository: {name} -> {url_or_path}")

    def discover_packs(self, repository_path: str) -> List[Dict[str, str]]:
        """Discover available packs in a repository"""
        logger.info(f"Discovering packs in repository: {repository_path}")

        discovered_packs = []
        repo_path = Path(repository_path)

        if repo_path.exists() and repo_path.is_dir():
            for item in repo_path.iterdir():
                if item.is_dir():
                    pack_manifest = item / "pack.yaml"
                    if pack_manifest.exists():
                        try:
                            metadata = self.pack_loader.load_pack_manifest(str(item))
                            discovered_packs.append({
                                "name": metadata.name,
                                "version": metadata.version,
                                "path": str(item),
                                "description": metadata.description
                            })
                            logger.info(f"  Discovered pack: {metadata.name} v{metadata.version}")
                        except Exception as e:
                            logger.warning(f"  Failed to load pack manifest from {item}: {e}")

        return discovered_packs

    def install_pack(self, pack_name: str, repository: str = None) -> bool:
        """Install a pack from a repository"""
        logger.info(f"Installing pack: {pack_name}")

        # If repository is specified, use it; otherwise search all repositories
        repositories_to_search = [repository] if repository else list(self.repositories.values())

        for repo_path in repositories_to_search:
            discovered_packs = self.discover_packs(repo_path)

            for pack_info in discovered_packs:
                if pack_info["name"] == pack_name:
                    # Load the pack
                    try:
                        metadata = self.pack_loader.load_pack(pack_info["path"])
                        self.pack_loader.register_pack_location(pack_name, pack_info["path"])

                        # Record installation
                        self.installed_packs[pack_name] = {
                            "metadata": metadata,
                            "installation_path": pack_info["path"],
                            "installed_at": datetime.now().isoformat(),
                            "repository": repo_path
                        }

                        logger.info(f"Successfully installed pack: {pack_name} v{metadata.version}")
                        return True

                    except Exception as e:
                        logger.error(f"Failed to install pack {pack_name}: {e}")
                        return False

        logger.error(f"Pack {pack_name} not found in any repository")
        return False

    def uninstall_pack(self, pack_name: str) -> bool:
        """Uninstall a pack"""
        logger.info(f"Uninstalling pack: {pack_name}")

        if pack_name in self.installed_packs:
            # Remove from installed packs
            del self.installed_packs[pack_name]

            # Remove from loaded packs
            if pack_name in self.pack_loader.loaded_packs:
                del self.pack_loader.loaded_packs[pack_name]

            # Remove from registry
            if pack_name in self.pack_loader.pack_registry:
                del self.pack_loader.pack_registry[pack_name]

            logger.info(f"Successfully uninstalled pack: {pack_name}")
            return True
        else:
            logger.warning(f"Pack {pack_name} is not installed")
            return False

    def list_installed_packs(self) -> List[Dict[str, Any]]:
        """List all installed packs"""
        return [
            {
                "name": pack_info["metadata"].name,
                "version": pack_info["metadata"].version,
                "description": pack_info["metadata"].description,
                "installed_at": pack_info["installed_at"],
                "path": pack_info["installation_path"]
            }
            for pack_info in self.installed_packs.values()
        ]

    def get_pack_details(self, pack_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an installed pack"""
        pack_info = self.installed_packs.get(pack_name)
        if pack_info:
            metadata = pack_info["metadata"]
            return {
                "name": metadata.name,
                "version": metadata.version,
                "kind": metadata.kind,
                "license": metadata.license,
                "description": metadata.description,
                "author": metadata.author,
                "tags": metadata.tags,
                "compatibility": metadata.compatibility,
                "dependencies": metadata.dependencies,
                "pipelines": metadata.pipelines,
                "agents": metadata.agents,
                "installation_info": {
                    "path": pack_info["installation_path"],
                    "installed_at": pack_info["installed_at"],
                    "repository": pack_info["repository"]
                }
            }
        return None


def demonstrate_pack_loading():
    """Demonstrate pack loading and management"""
    logger.info("=== Demonstrating Pack Loading and Management ===")

    # Create pack registry
    registry = PackRegistry()

    # Add repositories (using local examples directory)
    examples_path = "examples/packs"
    registry.add_repository("examples", examples_path)

    # Discover available packs
    logger.info("Discovering available packs...")
    available_packs = registry.discover_packs(examples_path)

    print("\nAvailable Packs:")
    print("-" * 50)
    for pack in available_packs:
        print(f"Name: {pack['name']}")
        print(f"Version: {pack['version']}")
        print(f"Description: {pack['description']}")
        print(f"Path: {pack['path']}")
        print("-" * 50)

    # Install packs
    logger.info("Installing packs...")
    for pack in available_packs:
        pack_name = pack["name"]
        success = registry.install_pack(pack_name)
        if success:
            logger.info(f"✓ Installed: {pack_name}")
        else:
            logger.error(f"✗ Failed to install: {pack_name}")

    # List installed packs
    installed = registry.list_installed_packs()
    print(f"\nInstalled Packs ({len(installed)}):")
    print("-" * 50)
    for pack in installed:
        print(f"• {pack['name']} v{pack['version']}")
        print(f"  {pack['description']}")
        print(f"  Installed: {pack['installed_at']}")
        print()

    return registry


def demonstrate_pipeline_execution(registry: PackRegistry):
    """Demonstrate executing pipelines from loaded packs"""
    logger.info("=== Demonstrating Pipeline Execution ===")

    installed_packs = registry.list_installed_packs()

    if not installed_packs:
        logger.warning("No packs installed for pipeline execution")
        return

    # Try to execute a pipeline from the first installed pack
    pack_info = installed_packs[0]
    pack_name = pack_info["name"]

    logger.info(f"Attempting to execute pipeline from pack: {pack_name}")

    # Get pack details
    pack_details = registry.get_pack_details(pack_name)
    if not pack_details:
        logger.error(f"Could not get details for pack: {pack_name}")
        return

    available_pipelines = pack_details["pipelines"]
    logger.info(f"Available pipelines: {available_pipelines}")

    if not available_pipelines:
        logger.warning(f"No pipelines available in pack: {pack_name}")
        return

    # Prepare sample inputs
    sample_inputs = {
        "facility_data": {
            "name": "SDK Example Facility",
            "location": "San Francisco, CA",
            "building_area_sqft": 75000,
            "electricity_kwh": 500000,
            "natural_gas_mmbtu": 3000
        },
        "analysis_config": {
            "include_scope3": False,
            "carbon_pricing": True,
            "benchmark_analysis": True
        }
    }

    # Execute the first available pipeline
    pipeline_name = available_pipelines[0]
    try:
        logger.info(f"Executing pipeline: {pipeline_name}")
        result = registry.pack_loader.execute_pack_pipeline(
            pack_name,
            pipeline_name,
            sample_inputs
        )

        print(f"\nPipeline Execution Result:")
        print("-" * 50)
        print(json.dumps(result, indent=2))

        return result

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return None


def demonstrate_pack_inspection():
    """Demonstrate pack inspection and metadata extraction"""
    logger.info("=== Demonstrating Pack Inspection ===")

    pack_loader = PackLoader()

    # Load the advanced pack example
    advanced_pack_path = "examples/packs/advanced"
    try:
        metadata = pack_loader.load_pack(advanced_pack_path)

        print("\nPack Inspection Results:")
        print("=" * 60)
        print(f"Name: {metadata.name}")
        print(f"Version: {metadata.version}")
        print(f"Kind: {metadata.kind}")
        print(f"License: {metadata.license}")
        print(f"Author: {metadata.author}")
        print(f"Description: {metadata.description}")
        print(f"\nTags: {', '.join(metadata.tags)}")
        print(f"\nCompatibility:")
        for key, value in metadata.compatibility.items():
            print(f"  {key}: {value}")
        print(f"\nDependencies:")
        for dep in metadata.dependencies:
            print(f"  • {dep}")
        print(f"\nPipelines:")
        for pipeline in metadata.pipelines:
            print(f"  • {pipeline}")
        print(f"\nAgents:")
        for agent in metadata.agents:
            print(f"  • {agent}")
        print("=" * 60)

        return metadata

    except Exception as e:
        logger.error(f"Pack inspection failed: {e}")
        return None


def create_pack_dependency_graph(registry: PackRegistry) -> Dict[str, List[str]]:
    """Create a dependency graph of installed packs"""
    logger.info("Creating pack dependency graph...")

    dependency_graph = {}
    installed_packs = registry.list_installed_packs()

    for pack_info in installed_packs:
        pack_name = pack_info["name"]
        pack_details = registry.get_pack_details(pack_name)

        if pack_details:
            dependencies = pack_details.get("dependencies", [])
            # Extract just the package names (remove version constraints)
            dep_names = []
            for dep in dependencies:
                if isinstance(dep, str):
                    # Extract package name (everything before version constraint)
                    dep_name = dep.split(">=")[0].split("==")[0].split(">")[0].split("<")[0]
                    dep_names.append(dep_name)

            dependency_graph[pack_name] = dep_names

    print("\nPack Dependency Graph:")
    print("-" * 40)
    for pack, deps in dependency_graph.items():
        print(f"{pack}:")
        if deps:
            for dep in deps:
                print(f"  └─ {dep}")
        else:
            print("  └─ (no dependencies)")
        print()

    return dependency_graph


def main():
    """Main function to run pack loading examples"""
    logger.info("Starting GreenLang SDK Pack Loading Examples...")

    try:
        # Demonstrate pack discovery and installation
        registry = demonstrate_pack_loading()

        # Demonstrate pack inspection
        pack_metadata = demonstrate_pack_inspection()

        # Demonstrate pipeline execution
        execution_result = demonstrate_pipeline_execution(registry)

        # Create dependency graph
        dependency_graph = create_pack_dependency_graph(registry)

        # Summary
        installed_packs = registry.list_installed_packs()
        print(f"\n=== Summary ===")
        print(f"Packs discovered and installed: {len(installed_packs)}")
        print(f"Pack inspection: {'✓ Success' if pack_metadata else '✗ Failed'}")
        print(f"Pipeline execution: {'✓ Success' if execution_result else '✗ Failed'}")
        print(f"Dependency analysis: ✓ Complete")

        logger.info("All pack loading examples completed successfully!")

        return {
            "registry": registry,
            "installed_packs": installed_packs,
            "pack_metadata": pack_metadata,
            "execution_result": execution_result,
            "dependency_graph": dependency_graph
        }

    except Exception as e:
        logger.error(f"Pack loading examples failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()