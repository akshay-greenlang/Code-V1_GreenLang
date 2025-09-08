"""
Pack Manifest Schema
====================

Defines the structure of pack.yaml files that describe domain packs.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from pathlib import Path
import yaml
from enum import Enum


class PackType(str, Enum):
    """Types of packs supported"""
    DOMAIN = "domain"       # Domain-specific logic (emissions, buildings, etc)
    CONNECTOR = "connector" # Data connectors (APIs, databases)
    REPORT = "report"      # Report templates and generators
    POLICY = "policy"      # Policy bundles
    DATASET = "dataset"    # Curated datasets


class Dependency(BaseModel):
    """Pack dependency specification"""
    name: str = Field(..., description="Pack name")
    version: str = Field(..., description="Version constraint (e.g., >=1.0.0)")
    registry: str = Field(default="hub.greenlang.io", description="Registry URL")


class Agent(BaseModel):
    """Agent export from a pack"""
    name: str = Field(..., description="Agent name")
    class_path: str = Field(..., description="Import path (e.g., agents.fuel:FuelAgent)")
    description: str = Field(..., description="What this agent does")
    inputs: Dict[str, str] = Field(default_factory=dict, description="Input schema")
    outputs: Dict[str, str] = Field(default_factory=dict, description="Output schema")


class Pipeline(BaseModel):
    """Pipeline export from a pack"""
    name: str = Field(..., description="Pipeline name")
    file: str = Field(..., description="YAML file path relative to pack root")
    description: str = Field(..., description="What this pipeline does")


class Dataset(BaseModel):
    """Dataset export from a pack"""
    name: str = Field(..., description="Dataset name")
    path: str = Field(..., description="Path relative to pack root")
    format: str = Field(..., description="Format (csv, json, parquet, etc)")
    card: Optional[str] = Field(None, description="Path to dataset card (markdown)")
    size: Optional[str] = Field(None, description="Dataset size")
    
    
class Model(BaseModel):
    """ML model export from a pack"""
    name: str = Field(..., description="Model name")
    path: str = Field(..., description="Path to model artifacts")
    framework: str = Field(..., description="Framework (sklearn, torch, tf, etc)")
    card: Optional[str] = Field(None, description="Path to model card")


class PackManifest(BaseModel):
    """
    Pack manifest (pack.yaml) schema
    
    Example:
    ```yaml
    name: emissions-core
    version: 1.0.0
    type: domain
    description: Core emissions calculation agents
    
    authors:
      - name: GreenLang Team
        email: team@greenlang.io
    
    dependencies:
      - name: greenlang-sdk
        version: ">=0.1.0"
    
    exports:
      agents:
        - name: FuelEmissions
          class_path: agents.fuel:FuelAgent
          description: Calculate fuel-based emissions
      
      pipelines:
        - name: building-analysis
          file: pipelines/building.yaml
          description: Complete building emissions analysis
      
      datasets:
        - name: emission-factors
          path: data/emission_factors.json
          format: json
          card: cards/emission_factors.md
    
    requirements:
      - pandas>=1.3.0
      - numpy>=1.20.0
    
    policy:
      install: policies/install.rego
      runtime: policies/runtime.rego
    
    provenance:
      sbom: true
      signing: true
    ```
    """
    
    # Basic metadata
    name: str = Field(..., description="Pack name (kebab-case)")
    version: str = Field(..., description="Semantic version")
    type: PackType = Field(PackType.DOMAIN, description="Pack type")
    description: str = Field(..., description="What this pack does")
    
    # Authors
    authors: List[Dict[str, str]] = Field(default_factory=list)
    license: str = Field(default="MIT", description="License")
    homepage: Optional[str] = Field(None, description="Project homepage")
    repository: Optional[str] = Field(None, description="Source repository")
    
    # Dependencies
    dependencies: List[Dependency] = Field(default_factory=list, description="Pack dependencies")
    requirements: List[str] = Field(default_factory=list, description="Python requirements")
    
    # Exports
    exports: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="What this pack exports (agents, pipelines, etc)"
    )
    
    # Entry points (optional)
    entry_points: Dict[str, str] = Field(
        default_factory=dict,
        description="Python entry points for registration"
    )
    
    # Policy hooks
    policy: Dict[str, str] = Field(
        default_factory=dict,
        description="OPA policy files for install/runtime"
    )
    
    # Provenance settings
    provenance: Dict[str, Any] = Field(
        default_factory=lambda: {"sbom": True, "signing": False},
        description="Provenance generation settings"
    )
    
    # Testing
    test_command: Optional[str] = Field(None, description="Command to run tests")
    
    # Minimum GreenLang version
    min_greenlang_version: str = Field(default="0.1.0")
    
    @validator('name')
    def validate_name(cls, v):
        """Ensure pack name is kebab-case"""
        import re
        if not re.match(r'^[a-z][a-z0-9-]*$', v):
            raise ValueError("Pack name must be kebab-case (lowercase with hyphens)")
        return v
    
    @classmethod
    def from_yaml(cls, path: Path) -> "PackManifest":
        """Load manifest from pack.yaml"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Path):
        """Save manifest to pack.yaml"""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(exclude_none=True), f, default_flow_style=False)
    
    def validate_structure(self, pack_dir: Path) -> List[str]:
        """Validate that pack directory matches manifest"""
        errors = []
        
        # Check required files exist
        if not (pack_dir / "pack.yaml").exists():
            errors.append("Missing pack.yaml")
        
        # Check exported agents exist
        if "agents" in self.exports:
            for agent in self.exports["agents"]:
                # Parse class_path to check file exists
                module_path = agent.get("class_path", "").split(":")[0].replace(".", "/")
                if module_path and not (pack_dir / f"{module_path}.py").exists():
                    errors.append(f"Agent module not found: {module_path}.py")
        
        # Check pipelines exist
        if "pipelines" in self.exports:
            for pipeline in self.exports["pipelines"]:
                if not (pack_dir / pipeline.get("file", "")).exists():
                    errors.append(f"Pipeline file not found: {pipeline.get('file')}")
        
        # Check datasets exist  
        if "datasets" in self.exports:
            for dataset in self.exports["datasets"]:
                if not (pack_dir / dataset.get("path", "")).exists():
                    errors.append(f"Dataset not found: {dataset.get('path')}")
        
        # Check policy files exist
        for policy_type, policy_file in self.policy.items():
            if not (pack_dir / policy_file).exists():
                errors.append(f"Policy file not found: {policy_file}")
        
        return errors