"""
Pack Loader
===========

Loads and initializes packs, making their components available.
"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .manifest import PackManifest
from .registry import PackRegistry, InstalledPack


logger = logging.getLogger(__name__)


class PackLoader:
    """
    Loads packs and makes their components available
    
    Handles:
    - Dynamic importing of pack modules
    - Agent registration
    - Pipeline loading
    - Dataset mounting
    - Policy injection
    """
    
    def __init__(self, registry: Optional[PackRegistry] = None):
        """
        Initialize loader
        
        Args:
            registry: Pack registry (creates default if not provided)
        """
        self.registry = registry or PackRegistry()
        self.loaded_packs: Dict[str, Any] = {}
        self.agents: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        self.datasets: Dict[str, Any] = {}
    
    def load(self, pack_name: str, verify: bool = True) -> Dict[str, Any]:
        """
        Load a pack and all its components
        
        Args:
            pack_name: Name of pack to load
            verify: Whether to verify pack before loading
        
        Returns:
            Dictionary of loaded components
        """
        # Check if already loaded
        if pack_name in self.loaded_packs:
            logger.info(f"Pack already loaded: {pack_name}")
            return self.loaded_packs[pack_name]
        
        # Get pack from registry
        pack = self.registry.get(pack_name)
        if not pack:
            raise ValueError(f"Pack not found: {pack_name}")
        
        # Verify if requested
        if verify and not pack.verified:
            if not self.registry.verify(pack_name):
                raise ValueError(f"Pack verification failed: {pack_name}")
        
        # Load based on location type
        if pack.location.startswith("entry_point:"):
            components = self._load_entry_point(pack)
        else:
            components = self._load_local_pack(pack)
        
        # Store loaded pack
        self.loaded_packs[pack_name] = components
        
        logger.info(f"Loaded pack: {pack_name}")
        return components
    
    def _load_entry_point(self, pack: InstalledPack) -> Dict[str, Any]:
        """Load pack from Python entry point"""
        ep_name = pack.location.replace("entry_point:", "")
        
        try:
            # Import the entry point module
            import importlib.metadata
            eps = importlib.metadata.entry_points()
            
            if hasattr(eps, 'select'):
                pack_eps = eps.select(group='greenlang.packs')
            else:
                pack_eps = eps.get('greenlang.packs', [])
            
            for ep in pack_eps:
                if ep.name == ep_name:
                    module = ep.load()
                    return self._extract_components(module, pack)
            
            raise ValueError(f"Entry point not found: {ep_name}")
            
        except Exception as e:
            logger.error(f"Failed to load entry point {ep_name}: {e}")
            raise
    
    def _load_local_pack(self, pack: InstalledPack) -> Dict[str, Any]:
        """Load pack from local filesystem"""
        pack_dir = Path(pack.location)
        
        # Load manifest
        manifest = PackManifest.from_yaml(pack_dir / "pack.yaml")
        
        # Add pack directory to Python path
        if str(pack_dir) not in sys.path:
            sys.path.insert(0, str(pack_dir))
        
        components = {
            "agents": {},
            "pipelines": {},
            "datasets": {},
            "models": {},
            "manifest": manifest
        }
        
        # Load agents
        if "agents" in manifest.exports:
            for agent_spec in manifest.exports["agents"]:
                agent = self._load_agent(pack_dir, agent_spec)
                if agent:
                    components["agents"][agent_spec["name"]] = agent
                    self.agents[f"{pack.name}.{agent_spec['name']}"] = agent
        
        # Load pipelines
        if "pipelines" in manifest.exports:
            for pipeline_spec in manifest.exports["pipelines"]:
                pipeline = self._load_pipeline(pack_dir, pipeline_spec)
                if pipeline:
                    components["pipelines"][pipeline_spec["name"]] = pipeline
                    self.pipelines[f"{pack.name}.{pipeline_spec['name']}"] = pipeline
        
        # Load datasets
        if "datasets" in manifest.exports:
            for dataset_spec in manifest.exports["datasets"]:
                dataset = self._load_dataset(pack_dir, dataset_spec)
                if dataset:
                    components["datasets"][dataset_spec["name"]] = dataset
                    self.datasets[f"{pack.name}.{dataset_spec['name']}"] = dataset
        
        return components
    
    def _load_agent(self, pack_dir: Path, agent_spec: Dict) -> Optional[Any]:
        """Load an agent from a pack"""
        try:
            class_path = agent_spec["class_path"]
            module_path, class_name = class_path.split(":")
            
            # Convert module path to file path
            module_file = pack_dir / f"{module_path.replace('.', '/')}.py"
            
            if not module_file.exists():
                logger.error(f"Agent module not found: {module_file}")
                return None
            
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(
                f"{pack_dir.name}.{module_path}",
                module_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the agent class
            agent_class = getattr(module, class_name)
            
            # Instantiate or return class based on agent design
            # For now, return the class itself
            return agent_class
            
        except Exception as e:
            logger.error(f"Failed to load agent {agent_spec['name']}: {e}")
            return None
    
    def _load_pipeline(self, pack_dir: Path, pipeline_spec: Dict) -> Optional[Any]:
        """Load a pipeline from a pack"""
        try:
            import yaml
            pipeline_file = pack_dir / pipeline_spec["file"]
            
            if not pipeline_file.exists():
                logger.error(f"Pipeline file not found: {pipeline_file}")
                return None
            
            with open(pipeline_file, 'r') as f:
                pipeline_data = yaml.safe_load(f)
            
            # Could instantiate a Pipeline object here
            # For now, return the raw data
            return pipeline_data
            
        except Exception as e:
            logger.error(f"Failed to load pipeline {pipeline_spec['name']}: {e}")
            return None
    
    def _load_dataset(self, pack_dir: Path, dataset_spec: Dict) -> Optional[Any]:
        """Load dataset metadata from a pack"""
        try:
            dataset_path = pack_dir / dataset_spec["path"]
            
            if not dataset_path.exists():
                logger.error(f"Dataset not found: {dataset_path}")
                return None
            
            # Load dataset card if available
            card = None
            if "card" in dataset_spec:
                card_path = pack_dir / dataset_spec["card"]
                if card_path.exists():
                    with open(card_path, 'r') as f:
                        card = f.read()
            
            # Return dataset metadata
            # Could load actual data here based on format
            return {
                "path": str(dataset_path),
                "format": dataset_spec.get("format"),
                "size": dataset_spec.get("size"),
                "card": card,
                "spec": dataset_spec
            }
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_spec['name']}: {e}")
            return None
    
    def _extract_components(self, module: Any, pack: InstalledPack) -> Dict[str, Any]:
        """Extract components from an imported module"""
        components = {
            "agents": {},
            "pipelines": {},
            "datasets": {},
            "models": {},
            "module": module
        }
        
        # Look for standard exports
        if hasattr(module, 'AGENTS'):
            for name, agent in module.AGENTS.items():
                components["agents"][name] = agent
                self.agents[f"{pack.name}.{name}"] = agent
        
        if hasattr(module, 'PIPELINES'):
            for name, pipeline in module.PIPELINES.items():
                components["pipelines"][name] = pipeline
                self.pipelines[f"{pack.name}.{name}"] = pipeline
        
        if hasattr(module, 'DATASETS'):
            for name, dataset in module.DATASETS.items():
                components["datasets"][name] = dataset
                self.datasets[f"{pack.name}.{name}"] = dataset
        
        return components
    
    def load_dependencies(self, pack_name: str):
        """Load all dependencies for a pack"""
        deps = self.registry.get_dependencies(pack_name)
        
        for dep in deps:
            # Parse dependency spec (name and version)
            if ">=" in dep:
                dep_name = dep.split(">=")[0]
            elif "==" in dep:
                dep_name = dep.split("==")[0]
            else:
                dep_name = dep
            
            # Load dependency if not already loaded
            if dep_name not in self.loaded_packs:
                try:
                    self.load(dep_name)
                except Exception as e:
                    logger.warning(f"Failed to load dependency {dep_name}: {e}")
    
    def get_agent(self, agent_ref: str) -> Optional[Any]:
        """
        Get an agent by reference
        
        Args:
            agent_ref: Agent reference (pack.agent or just agent)
        
        Returns:
            Agent class or instance
        """
        # Try direct lookup first
        if agent_ref in self.agents:
            return self.agents[agent_ref]
        
        # Try without pack prefix
        for key, agent in self.agents.items():
            if key.endswith(f".{agent_ref}"):
                return agent
        
        return None
    
    def get_pipeline(self, pipeline_ref: str) -> Optional[Any]:
        """Get a pipeline by reference"""
        if pipeline_ref in self.pipelines:
            return self.pipelines[pipeline_ref]
        
        for key, pipeline in self.pipelines.items():
            if key.endswith(f".{pipeline_ref}"):
                return pipeline
        
        return None
    
    def get_dataset(self, dataset_ref: str) -> Optional[Any]:
        """Get a dataset by reference"""
        if dataset_ref in self.datasets:
            return self.datasets[dataset_ref]
        
        for key, dataset in self.datasets.items():
            if key.endswith(f".{dataset_ref}"):
                return dataset
        
        return None
    
    def list_agents(self) -> List[str]:
        """List all available agents"""
        return list(self.agents.keys())
    
    def list_pipelines(self) -> List[str]:
        """List all available pipelines"""
        return list(self.pipelines.keys())
    
    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        return list(self.datasets.keys())