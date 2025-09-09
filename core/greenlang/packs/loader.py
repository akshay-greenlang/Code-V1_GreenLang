"""
Pack Loader
===========

Loads and initializes packs from various sources:
- Entry points (pip-installed packs)
- Local directories
- Pack archives (.glpack files)
"""

import importlib
import importlib.util
import importlib.metadata as md
import logging
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import yaml

from .manifest import PackManifest, load_manifest
from .registry import InstalledPack

logger = logging.getLogger(__name__)

# Entry point group for discovering installed packs
ENTRY_GROUP = "greenlang.packs"


class PackLoader:
    """
    Loads packs and makes their components available
    
    Supports:
    - Entry points (pip-installed packs)
    - Local directories
    - Pack archives (.glpack)
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize loader
        
        Args:
            cache_dir: Directory for caching extracted packs
        """
        self.cache_dir = cache_dir or Path.home() / ".greenlang" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaded_packs: Dict[str, LoadedPack] = {}
        self.discovered_packs: Dict[str, PackManifest] = {}
        
        # Discover available packs
        self._discover_all()
    
    def _discover_all(self):
        """Discover all available packs from various sources"""
        # Discover from entry points
        self.discovered_packs.update(discover_installed())
        
        # Discover from standard local directories
        local_dirs = [
            Path.cwd() / "packs",
            Path.home() / ".greenlang" / "packs",
        ]
        
        for dir_path in local_dirs:
            if dir_path.exists():
                self.discovered_packs.update(discover_local_packs(dir_path))
    
    def load(self, pack_ref: str, verify: bool = True) -> 'LoadedPack':
        """
        Load a pack by reference
        
        Args:
            pack_ref: Pack reference (name, path, or name@version)
            verify: Whether to verify pack integrity
        
        Returns:
            LoadedPack instance
        """
        # Check if already loaded
        if pack_ref in self.loaded_packs:
            logger.info(f"Pack already loaded: {pack_ref}")
            return self.loaded_packs[pack_ref]
        
        # Parse pack reference
        pack_name, version = parse_pack_ref(pack_ref)
        
        # Try to find pack
        pack_path = self._resolve_pack_path(pack_name, version)
        
        if not pack_path:
            raise ValueError(f"Pack not found: {pack_ref}")
        
        # Load the pack
        loaded_pack = self._load_from_path(pack_path, verify)
        
        # Cache it
        self.loaded_packs[pack_name] = loaded_pack
        
        logger.info(f"Loaded pack: {pack_name} from {pack_path}")
        return loaded_pack
    
    def _resolve_pack_path(self, pack_name: str, version: Optional[str] = None) -> Optional[Path]:
        """
        Resolve pack name to path
        
        Args:
            pack_name: Pack name
            version: Optional version constraint
        
        Returns:
            Path to pack directory or None if not found
        """
        # Check discovered packs
        if pack_name in self.discovered_packs:
            manifest = self.discovered_packs[pack_name]
            
            # Check version if specified
            if version and not version_matches(manifest.version, version):
                logger.warning(f"Version mismatch: {pack_name} {manifest.version} vs {version}")
                return None
            
            # Get pack location
            if hasattr(manifest, '_location'):
                return Path(manifest._location)
        
        # Check if it's a direct path
        if Path(pack_name).exists():
            return Path(pack_name)
        
        # Check cache for extracted packs
        cached_path = self.cache_dir / pack_name
        if cached_path.exists():
            return cached_path
        
        return None
    
    def _load_from_path(self, pack_path: Path, verify: bool = True) -> 'LoadedPack':
        """
        Load pack from a directory path
        
        Args:
            pack_path: Path to pack directory
            verify: Whether to verify pack
        
        Returns:
            LoadedPack instance
        """
        # Load manifest
        manifest = load_manifest(pack_path)
        
        # Create loaded pack
        loaded_pack = LoadedPack(
            manifest=manifest,
            path=pack_path,
            loader=self
        )
        
        # Load components
        loaded_pack.load_components()
        
        return loaded_pack
    
    def load_from_archive(self, archive_path: Path) -> 'LoadedPack':
        """
        Load pack from .glpack archive
        
        Args:
            archive_path: Path to .glpack file
        
        Returns:
            LoadedPack instance
        """
        if not archive_path.exists():
            raise ValueError(f"Archive not found: {archive_path}")
        
        # Extract to cache
        extract_dir = self.cache_dir / archive_path.stem
        
        if not extract_dir.exists():
            logger.info(f"Extracting {archive_path} to {extract_dir}")
            
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_dir)
        
        # Find pack directory (might be nested)
        pack_dirs = list(extract_dir.glob("*/pack.yaml"))
        if pack_dirs:
            pack_dir = pack_dirs[0].parent
        else:
            pack_dir = extract_dir
        
        return self._load_from_path(pack_dir)
    
    def list_available(self) -> List[str]:
        """List all available pack names"""
        return list(self.discovered_packs.keys())
    
    def get_manifest(self, pack_name: str) -> Optional[PackManifest]:
        """Get manifest for a pack without loading it"""
        return self.discovered_packs.get(pack_name)


class LoadedPack:
    """
    Represents a loaded pack with all its components
    """
    
    def __init__(self, manifest: PackManifest, path: Path, loader: PackLoader):
        """
        Initialize loaded pack
        
        Args:
            manifest: Pack manifest
            path: Path to pack directory
            loader: Parent loader instance
        """
        self.manifest = manifest
        self.path = path
        self.loader = loader
        
        # Component storage
        self.agents: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        self.datasets: Dict[str, Any] = {}
        self.reports: Dict[str, Any] = {}
        self.connectors: Dict[str, Any] = {}
        
        # Module cache
        self._modules: Dict[str, Any] = {}
    
    def load_components(self):
        """Load all pack components"""
        # Add pack directory to Python path if needed
        if str(self.path) not in sys.path:
            sys.path.insert(0, str(self.path))
        
        try:
            # Load agents
            self._load_agents()
            
            # Load pipelines
            self._load_pipelines()
            
            # Load datasets
            self._load_datasets()
            
            # Load reports
            self._load_reports()
            
        finally:
            # Remove from path to avoid conflicts
            if str(self.path) in sys.path:
                sys.path.remove(str(self.path))
    
    def _load_agents(self):
        """Load agent classes"""
        for agent_name in self.manifest.contents.agents:
            try:
                # Try to import from agents module
                agent_module_path = self.path / "agents"
                
                if agent_module_path.exists():
                    # Import the module
                    module_name = f"{self.manifest.name}.agents"
                    spec = importlib.util.spec_from_file_location(
                        module_name,
                        agent_module_path / "__init__.py"
                    )
                    
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Try to get the agent class
                        if hasattr(module, agent_name):
                            self.agents[agent_name] = getattr(module, agent_name)
                            logger.info(f"Loaded agent: {agent_name}")
                
            except Exception as e:
                logger.error(f"Failed to load agent {agent_name}: {e}")
    
    def _load_pipelines(self):
        """Load pipeline definitions"""
        for pipeline_file in self.manifest.contents.pipelines:
            try:
                pipeline_path = self.path / pipeline_file
                
                if pipeline_path.exists():
                    with open(pipeline_path) as f:
                        pipeline_data = yaml.safe_load(f)
                    
                    # Store pipeline data
                    pipeline_name = pipeline_path.stem
                    self.pipelines[pipeline_name] = pipeline_data
                    logger.info(f"Loaded pipeline: {pipeline_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load pipeline {pipeline_file}: {e}")
    
    def _load_datasets(self):
        """Load dataset metadata"""
        for dataset_name in self.manifest.contents.datasets:
            try:
                dataset_path = self.path / "datasets" / dataset_name
                
                if dataset_path.exists():
                    # Load dataset metadata
                    dataset_info = {
                        "path": str(dataset_path),
                        "name": dataset_name,
                        "format": dataset_path.suffix[1:],  # Remove leading dot
                        "size": dataset_path.stat().st_size
                    }
                    
                    # Check for dataset card
                    card_name = dataset_path.stem + ".md"
                    card_path = self.path / "cards" / card_name
                    
                    if card_path.exists():
                        with open(card_path) as f:
                            dataset_info["card"] = f.read()
                    
                    self.datasets[dataset_name] = dataset_info
                    logger.info(f"Loaded dataset: {dataset_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
    
    def _load_reports(self):
        """Load report templates"""
        for report_name in self.manifest.contents.reports:
            try:
                report_path = self.path / "reports" / report_name
                
                if report_path.exists():
                    with open(report_path) as f:
                        template_content = f.read()
                    
                    self.reports[report_name] = {
                        "path": str(report_path),
                        "template": template_content,
                        "name": report_name
                    }
                    logger.info(f"Loaded report template: {report_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load report {report_name}: {e}")
    
    def get_agent(self, agent_name: str) -> Optional[Any]:
        """Get an agent class by name"""
        return self.agents.get(agent_name)
    
    def get_pipeline(self, pipeline_name: str) -> Optional[Dict]:
        """Get a pipeline definition by name"""
        return self.pipelines.get(pipeline_name)
    
    def get_dataset(self, dataset_name: str) -> Optional[Dict]:
        """Get dataset metadata by name"""
        return self.datasets.get(dataset_name)
    
    def get_report(self, report_name: str) -> Optional[Dict]:
        """Get report template by name"""
        return self.reports.get(report_name)


def discover_installed() -> Dict[str, PackManifest]:
    """
    Discover packs installed via pip (entry points)
    
    Returns:
        Dictionary of pack name to manifest
    """
    found = {}
    
    try:
        # Get entry points for our group
        if sys.version_info >= (3, 10):
            eps = md.entry_points(group=ENTRY_GROUP)
        else:
            eps = md.entry_points().get(ENTRY_GROUP, [])
        
        for ep in eps:
            try:
                # Load the entry point - should return path to pack.yaml
                manifest_path_func = ep.load()
                
                if callable(manifest_path_func):
                    manifest_path = Path(manifest_path_func())
                else:
                    manifest_path = Path(manifest_path)
                
                if manifest_path.exists():
                    # Load manifest
                    manifest = PackManifest.from_yaml(manifest_path.parent)
                    
                    # Store location for later use
                    manifest._location = str(manifest_path.parent)
                    
                    found[manifest.name] = manifest
                    logger.info(f"Discovered installed pack: {manifest.name} v{manifest.version}")
                    
            except Exception as e:
                logger.error(f"Failed to load entry point {ep.name}: {e}")
                
    except Exception as e:
        logger.debug(f"No entry points found: {e}")
    
    return found


def discover_local_packs(base_dir: Path) -> Dict[str, PackManifest]:
    """
    Discover packs in a local directory
    
    Args:
        base_dir: Directory to search for packs
    
    Returns:
        Dictionary of pack name to manifest
    """
    found = {}
    
    if not base_dir.exists():
        return found
    
    # Look for pack.yaml files
    for pack_yaml in base_dir.glob("*/pack.yaml"):
        try:
            pack_dir = pack_yaml.parent
            manifest = PackManifest.from_yaml(pack_dir)
            
            # Store location
            manifest._location = str(pack_dir)
            
            found[manifest.name] = manifest
            logger.info(f"Discovered local pack: {manifest.name} v{manifest.version}")
            
        except Exception as e:
            logger.error(f"Failed to load pack from {pack_yaml}: {e}")
    
    return found


def load_from_path(path: str) -> PackManifest:
    """
    Load a pack manifest from a path
    
    Args:
        path: Path to pack directory
    
    Returns:
        PackManifest instance
    """
    p = Path(path)
    
    if p.is_file() and p.name == "pack.yaml":
        manifest_path = p
    else:
        manifest_path = p / "pack.yaml"
    
    if not manifest_path.exists():
        raise ValueError(f"No pack.yaml found at {path}")
    
    return PackManifest.from_yaml(manifest_path.parent)


def parse_pack_ref(pack_ref: str) -> tuple[str, Optional[str]]:
    """
    Parse a pack reference
    
    Args:
        pack_ref: Pack reference (e.g., "name", "name@1.0.0", "name>=1.0")
    
    Returns:
        Tuple of (name, version_constraint)
    """
    if "@" in pack_ref:
        name, version = pack_ref.split("@", 1)
        return name, version
    elif any(op in pack_ref for op in [">=", "<=", "==", ">", "<"]):
        # Find the operator
        for op in [">=", "<=", "==", ">", "<"]:
            if op in pack_ref:
                name, version = pack_ref.split(op, 1)
                return name.strip(), op + version.strip()
    
    return pack_ref, None


def version_matches(actual: str, constraint: str) -> bool:
    """
    Check if version matches constraint
    
    Args:
        actual: Actual version
        constraint: Version constraint
    
    Returns:
        True if version matches
    """
    # Simple implementation - in production use packaging.version
    if not constraint:
        return True
    
    if constraint.startswith(">="):
        required = constraint[2:]
        return actual >= required
    elif constraint.startswith("=="):
        required = constraint[2:]
        return actual == required
    
    # Default: accept
    return True