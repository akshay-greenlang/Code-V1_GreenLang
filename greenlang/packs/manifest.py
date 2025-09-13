"""
Pack Manifest v1.0 - Pydantic Models for GreenLang Pack Specification
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Union, Literal, Dict, Any
import re
from pathlib import Path


class Compat(BaseModel):
    """Compatibility constraints for runtime environments"""
    greenlang: Optional[str] = Field(None, description="GreenLang version compatibility range")
    python: Optional[str] = Field(None, description="Python version compatibility range")


class Contents(BaseModel):
    """Contents and artifacts provided by this pack"""
    pipelines: List[str] = Field(..., min_length=1, description="List of pipeline configuration files")
    agents: List[str] = Field(default_factory=list, description="List of agent names")
    datasets: List[str] = Field(default_factory=list, description="Dataset file paths")
    reports: List[str] = Field(default_factory=list, description="Report template paths")
    
    @field_validator('pipelines')
    @classmethod
    def validate_pipelines(cls, v: List[str]) -> List[str]:
        """Ensure at least one pipeline is defined"""
        if not v or len(v) == 0:
            raise ValueError("At least one pipeline must be defined in contents.pipelines")
        return v


class Security(BaseModel):
    """Security metadata and constraints"""
    sbom: Optional[str] = Field(None, description="Path to Software Bill of Materials file")
    signatures: List[str] = Field(default_factory=list, description="Digital signature files")
    vulnerabilities: Optional[Dict[str, Any]] = Field(None, description="Vulnerability tolerance settings")


class PackManifest(BaseModel):
    """
    GreenLang Pack Manifest v1.0
    
    This is the canonical schema for pack.yaml files.
    All packs must conform to this specification.
    """
    
    # Required fields
    name: str = Field(..., description="DNS-safe pack name")
    version: str = Field(..., description="Semantic version in MAJOR.MINOR.PATCH format")
    kind: Literal["pack", "dataset", "connector"] = Field("pack", description="Type of GreenLang package")
    license: str = Field(..., description="SPDX license identifier")
    contents: Contents = Field(..., description="Pack contents and artifacts")
    
    # Optional fields
    compat: Optional[Compat] = Field(None, description="Compatibility constraints")
    dependencies: List[Union[str, Dict[str, str]]] = Field(
        default_factory=list,
        description="External dependencies required by this pack"
    )
    card: Optional[str] = Field(None, description="Path to Model Card or Pack Card documentation")
    policy: Dict[str, Any] = Field(
        default_factory=dict,
        description="Policy constraints and requirements"
    )
    security: Security = Field(
        default_factory=Security,
        description="Security metadata and constraints"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for pack discovery"
    )
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate pack name is DNS-safe"""
        pattern = r"^[a-z][a-z0-9-]{1,62}[a-z0-9]$"
        if not re.match(pattern, v):
            raise ValueError(
                f"Pack name '{v}' must be DNS-safe: lowercase letters, numbers, and hyphens only, "
                f"3-64 characters, must start with letter and end with alphanumeric"
            )
        return v
    
    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format"""
        pattern = r"^\d+\.\d+\.\d+$"
        if not re.match(pattern, v):
            raise ValueError(
                f"Version '{v}' must be semantic version in MAJOR.MINOR.PATCH format (e.g., 1.0.0)"
            )
        return v
    
    @field_validator("license")
    @classmethod
    def validate_license(cls, v: str) -> str:
        """Validate SPDX license identifier"""
        # Common SPDX licenses - this should be strict for v1.0
        valid_licenses = {
            "MIT", "Apache-2.0", "GPL-3.0", "GPL-2.0", "BSD-3-Clause", "BSD-2-Clause",
            "ISC", "MPL-2.0", "LGPL-3.0", "LGPL-2.1", "AGPL-3.0", "Unlicense",
            "CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "EPL-2.0", "EPL-1.0",
            "AFL-3.0", "Artistic-2.0", "BSL-1.0", "CECILL-2.1", "ECL-2.0",
            "EUPL-1.2", "GPL-3.0-only", "GPL-2.0-only", "LGPL-3.0-only", "LGPL-2.1-only",
            "MIT-0", "MS-PL", "MS-RL", "NCSA", "OFL-1.1", "OSL-3.0", "PostgreSQL",
            "Zlib", "0BSD", "BSD-4-Clause", "BSD-3-Clause-Clear", "WTFPL",
            "Commercial", "Proprietary", "UNLICENSED"
        }
        
        # Strict validation for v1.0
        if v not in valid_licenses:
            raise ValueError(
                f"License '{v}' is not a valid SPDX identifier. "
                f"Use one of: MIT, Apache-2.0, GPL-3.0, BSD-3-Clause, ISC, MPL-2.0, etc. "
                f"See https://spdx.org/licenses/ for full list."
            )
        
        return v
    
    @model_validator(mode='after')
    def validate_file_paths(self) -> 'PackManifest':
        """
        Validate that referenced files exist (when running in pack directory).
        This is a post-validation check that can be optionally enforced.
        """
        # This validation is context-dependent and would be called
        # explicitly by the CLI validator when checking a real pack
        return self
    
    def validate_files_exist(self, base_path: Path) -> List[str]:
        """
        Check that all referenced files exist relative to base_path.
        Returns list of missing files.
        """
        missing = []
        base_path = Path(base_path)
        
        # Check pipeline files
        for pipeline in self.contents.pipelines:
            if not (base_path / pipeline).exists():
                missing.append(f"Pipeline file not found: {pipeline}")
        
        # Check dataset files
        for dataset in self.contents.datasets:
            if not (base_path / dataset).exists():
                missing.append(f"Dataset file not found: {dataset}")
        
        # Check report templates
        for report in self.contents.reports:
            if not (base_path / report).exists():
                missing.append(f"Report template not found: {report}")
        
        # Check card if specified
        if self.card and not (base_path / self.card).exists():
            missing.append(f"Card file not found: {self.card}")
        
        # Check SBOM if specified
        if self.security.sbom and not (base_path / self.security.sbom).exists():
            missing.append(f"SBOM file not found: {self.security.sbom}")
        
        # Check signatures
        for sig in self.security.signatures:
            if not (base_path / sig).exists():
                missing.append(f"Signature file not found: {sig}")
        
        return missing
    
    def get_warnings(self) -> List[str]:
        """
        Get list of warnings for recommended but missing fields.
        """
        warnings = []
        
        if not self.card:
            warnings.append("Recommended: Add 'card' field pointing to CARD.md or README.md")
        
        if not self.compat:
            warnings.append("Recommended: Add 'compat' field to specify version requirements")
        
        if self.security and not self.security.sbom:
            warnings.append("Recommended: Add 'security.sbom' for supply chain security")
        
        if not self.metadata or 'description' not in self.metadata:
            warnings.append("Recommended: Add 'metadata.description' for pack discovery")
        
        # Check for version constraints in dependencies
        for dep in self.dependencies:
            if isinstance(dep, str) and not any(op in dep for op in ['>=', '<=', '==', '>', '<', '~=', '^']):
                warnings.append(f"Recommended: Add version constraint to dependency '{dep}'")
        
        return warnings
    
    def to_yaml(self) -> str:
        """Export manifest to YAML format"""
        import yaml
        data = self.model_dump(exclude_none=True, exclude_defaults=False)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    def to_json(self) -> str:
        """Export manifest to JSON format"""
        import json
        data = self.model_dump(exclude_none=True, exclude_defaults=False)
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_yaml(cls, yaml_content: str) -> 'PackManifest':
        """Create manifest from YAML content"""
        import yaml
        data = yaml.safe_load(yaml_content)
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_content: str) -> 'PackManifest':
        """Create manifest from JSON content"""
        import json
        data = json.loads(json_content)
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'PackManifest':
        """Load manifest from file (YAML or JSON)"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {file_path}")
        
        content = file_path.read_text()
        
        if file_path.suffix in ['.yaml', '.yml']:
            return cls.from_yaml(content)
        elif file_path.suffix == '.json':
            return cls.from_json(content)
        else:
            # Try YAML first, then JSON
            try:
                return cls.from_yaml(content)
            except Exception:
                return cls.from_json(content)


# Backward compatibility aliases
PackManifestV1 = PackManifest