"""
Pack Manifest Schema
====================

Defines the structure and validation for pack.yaml files.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal, Optional, Dict, Any, Union
from pathlib import Path
import yaml
import re


class Compat(BaseModel):
    """Compatibility requirements"""

    greenlang: str = Field(
        ..., description="GreenLang version constraint (e.g., '>=0.1,<0.4')"
    )
    python: str = Field(..., description="Python version constraint (e.g., '>=3.10')")

    @field_validator("greenlang", "python")
    @classmethod
    def validate_version_spec(cls, v: str) -> str:
        """Validate version specifications"""
        # Basic validation - should have comparison operator
        if not any(op in v for op in [">=", "<=", "==", ">", "<", "~=", "!="]):
            raise ValueError(f"Invalid version spec: {v}")
        return v


class Policy(BaseModel):
    """Policy constraints for the pack"""

    network: List[str] = Field(
        default_factory=list,
        description="Network egress allowlist (e.g., ['era5:*', 'api.weather.gov'])",
    )
    data_residency: List[str] = Field(
        default_factory=list,
        description="Allowed data residency regions (e.g., ['IN', 'EU', 'US'])",
    )
    ef_vintage_min: Optional[int] = Field(
        None, description="Minimum emission factor vintage year"
    )
    license_allowlist: List[str] = Field(
        default=["Apache-2.0", "MIT", "Commercial"],
        description="Allowed licenses for dependencies",
    )
    max_memory_mb: Optional[int] = Field(None, description="Maximum memory usage in MB")
    max_cpu_cores: Optional[float] = Field(None, description="Maximum CPU cores")

    @field_validator("ef_vintage_min")
    @classmethod
    def validate_vintage(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 2020:
            raise ValueError("Emission factor vintage must be 2020 or later")
        return v


class Security(BaseModel):
    """Security and provenance settings"""

    sbom: Optional[str] = Field(
        None, description="Path to SBOM file (e.g., 'sbom.spdx.json')"
    )
    signatures: List[str] = Field(
        default_factory=list, description="List of signature files"
    )
    keyless: bool = Field(default=False, description="Use keyless signing (sigstore)")
    attestations: List[str] = Field(
        default_factory=list, description="Additional attestations (SLSA, etc.)"
    )


class Contents(BaseModel):
    """Pack contents declaration"""

    pipelines: List[str] = Field(
        default_factory=list, description="Pipeline YAML files (e.g., ['gl.yaml'])"
    )
    agents: List[str] = Field(
        default_factory=list, description="Agent names exported by this pack"
    )
    datasets: List[str] = Field(
        default_factory=list, description="Dataset files included"
    )
    reports: List[str] = Field(
        default_factory=list, description="Report templates (Jinja2)"
    )
    models: List[str] = Field(default_factory=list, description="ML models included")
    connectors: List[str] = Field(
        default_factory=list, description="External connectors"
    )


class Metadata(BaseModel):
    """Pack metadata"""

    authors: List[Dict[str, str]] = Field(
        default_factory=list, description="Pack authors"
    )
    homepage: Optional[str] = Field(None, description="Project homepage")
    repository: Optional[str] = Field(None, description="Source repository")
    documentation: Optional[str] = Field(None, description="Documentation URL")
    keywords: List[str] = Field(default_factory=list, description="Search keywords")


class PackManifest(BaseModel):
    """Complete pack manifest (pack.yaml) schema"""

    # Required fields
    name: str = Field(..., description="Pack name (kebab-case)")
    version: str = Field(..., description="Semantic version")
    kind: Literal["pack", "dataset", "connector"] = Field(
        "pack", description="Pack type"
    )
    license: str = Field(..., description="License identifier")

    # Now optional fields (was required before)
    compat: Optional[Compat] = Field(None, description="Compatibility requirements")
    contents: Optional[Contents] = Field(None, description="Pack contents")
    card: Optional[str] = Field(None, description="Path to pack card (CARD.md)")

    # Optional fields
    description: Optional[str] = Field(None, description="Short description")
    policy: Policy = Field(default_factory=Policy, description="Policy constraints")
    security: Security = Field(
        default_factory=Security, description="Security settings"
    )
    tests: List[str] = Field(default_factory=list, description="Test files")
    metadata: Optional[Metadata] = Field(None, description="Additional metadata")
    dependencies: List[Union[str, Dict[str, str]]] = Field(
        default_factory=list, description="Other packs this pack depends on"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure pack name is kebab-case"""
        if not re.match(r"^[a-z][a-z0-9-]*$", v):
            raise ValueError("Pack name must be kebab-case (lowercase with hyphens)")
        return v

    @field_validator("version")
    @classmethod
    def semver_ok(cls, v: str) -> str:
        """Validate semantic versioning"""
        # Simple validation for semantic versioning
        if not v or "." not in v:
            raise ValueError(
                f"Version must be in semantic versioning format (e.g., 1.0.0)"
            )

        parts = v.split(".")
        if len(parts) < 2:
            raise ValueError(f"Version must have at least major.minor: {v}")

        # Check that main parts are numeric
        for i, part in enumerate(parts[:3]):  # Check major, minor, patch
            if i < len(parts):
                # Allow pre-release and build metadata after patch
                base_part = part.split("-")[0].split("+")[0]
                if not base_part.isdigit():
                    raise ValueError(f"Invalid version component '{part}' in: {v}")

        return v

    @field_validator("license")
    @classmethod
    def validate_license(cls, v: str) -> str:
        """Validate license identifier"""
        # Common SPDX identifiers + custom
        valid_licenses = [
            "MIT",
            "Apache-2.0",
            "GPL-3.0",
            "BSD-3-Clause",
            "ISC",
            "MPL-2.0",
            "LGPL-3.0",
            "Commercial",
            "Proprietary",
        ]
        if v not in valid_licenses and not v.startswith("LicenseRef-"):
            raise ValueError(
                f"Unknown license: {v}. Use SPDX identifier or LicenseRef-*"
            )
        return v

    @model_validator(mode="after")
    def validate_contents(self) -> "PackManifest":
        """Validate that contents has at least something"""
        # Only validate if contents is provided
        if self.contents:
            c = self.contents
            if not any(
                [c.pipelines, c.agents, c.datasets, c.reports, c.models, c.connectors]
            ):
                raise ValueError(
                    "Pack must contain at least one pipeline, agent, dataset, report, model, or connector"
                )
        return self

    def validate_files(self, pack_dir: Path) -> List[str]:
        """
        Validate that all referenced files exist

        Args:
            pack_dir: Root directory of the pack

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check card exists (if specified)
        if self.card and not (pack_dir / self.card).exists():
            errors.append(f"Card file not found: {self.card}")

        # Check contents (if specified)
        if self.contents:
            # Check pipelines
            for pipeline in self.contents.pipelines:
                if not (pack_dir / pipeline).exists():
                    errors.append(f"Pipeline not found: {pipeline}")

            # Check datasets
            for dataset in self.contents.datasets:
                # Dataset path may already include 'datasets/' prefix
                if dataset.startswith("datasets/"):
                    dataset_path = pack_dir / dataset
                else:
                    dataset_path = pack_dir / "datasets" / dataset
                if not dataset_path.exists():
                    errors.append(f"Dataset not found: {dataset}")

            # Check reports
            for report in self.contents.reports:
                report_path = pack_dir / "reports" / report
                if not report_path.exists():
                    errors.append(f"Report template not found: reports/{report}")

        # Check tests
        for test in self.tests:
            # Handle wildcards in test patterns
            if "*" in test:
                # Convert to Path and use glob
                test_pattern = pack_dir / test
                matching_files = list(pack_dir.glob(test))
                if not matching_files:
                    errors.append(f"No test files matching pattern: {test}")
            else:
                if not (pack_dir / test).exists():
                    errors.append(f"Test file not found: {test}")

        # Check SBOM if specified
        if self.security.sbom:
            if not (pack_dir / self.security.sbom).exists():
                errors.append(f"SBOM file not found: {self.security.sbom}")

        # Check signatures
        for sig in self.security.signatures:
            if not (pack_dir / sig).exists():
                errors.append(f"Signature file not found: {sig}")

        return errors

    @classmethod
    def from_yaml(cls, path: Path) -> "PackManifest":
        """Load manifest from pack.yaml"""
        yaml_path = path / "pack.yaml" if path.is_dir() else path

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: Path):
        """Save manifest to pack.yaml"""
        yaml_path = path / "pack.yaml" if path.is_dir() else path

        # Convert to dict, excluding None values
        data = self.model_dump(exclude_none=True, exclude_defaults=False)

        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_json_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for validation"""
        return self.model_json_schema()


def load_manifest(path: Path) -> PackManifest:
    """
    Load and validate a pack manifest

    Args:
        path: Path to pack directory or pack.yaml file

    Returns:
        Validated PackManifest

    Raises:
        ValueError: If manifest is invalid
    """
    # Load manifest
    manifest = PackManifest.from_yaml(path)

    # Determine pack directory
    pack_dir = path if path.is_dir() else path.parent

    # Validate files exist
    errors = manifest.validate_files(pack_dir)
    if errors:
        raise ValueError(
            f"Pack validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return manifest


def validate_pack(pack_dir: Path) -> tuple[bool, List[str]]:
    """
    Validate a pack directory

    Args:
        pack_dir: Path to pack directory

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check pack.yaml exists
    manifest_path = pack_dir / "pack.yaml"
    if not manifest_path.exists():
        errors.append("No pack.yaml found")
        return False, errors

    try:
        # Load and validate manifest
        manifest = load_manifest(pack_dir)

        # Additional validation
        if manifest.policy.ef_vintage_min:
            if manifest.policy.ef_vintage_min < 2024:
                errors.append(
                    f"Emission factor vintage {manifest.policy.ef_vintage_min} is too old (min: 2024)"
                )

        # Check network policy - warn but don't fail if empty
        # (empty means no external calls allowed, which is valid)
        # Only fail if network policy is missing entirely
        pass  # Network policy can be empty

        # License check
        if manifest.license not in manifest.policy.license_allowlist:
            errors.append(
                f"License {manifest.license} not in allowlist: {manifest.policy.license_allowlist}"
            )

    except Exception as e:
        errors.append(f"Manifest validation failed: {e}")

    return len(errors) == 0, errors


def create_pack_template(pack_dir: Path, name: str, kind: str = "pack"):
    """
    Create a new pack with template structure

    Args:
        pack_dir: Directory to create pack in
        name: Pack name
        kind: Pack type
    """
    pack_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest = PackManifest(
        name=name,
        version="0.1.0",
        kind=kind,
        license="MIT",
        description=f"A {kind} pack for {name}",
        compat=Compat(greenlang=">=0.1", python=">=3.10"),
        contents=Contents(
            pipelines=["gl.yaml"] if kind == "pack" else [],
            agents=[],
            datasets=[],
            reports=[],
        ),
        policy=Policy(
            network=["*"] if kind == "connector" else [],
            data_residency=["US", "EU"],
            ef_vintage_min=2024,
        ),
        security=Security(sbom="sbom.spdx.json"),
        tests=["tests/test_pipeline.py"] if kind == "pack" else [],
        card="CARD.md",
    )

    manifest.to_yaml(pack_dir)

    # Create directories
    (pack_dir / "datasets").mkdir(exist_ok=True)
    (pack_dir / "reports").mkdir(exist_ok=True)
    (pack_dir / "tests").mkdir(exist_ok=True)
    (pack_dir / "tests" / "golden").mkdir(exist_ok=True)

    # Create template files
    (pack_dir / "CARD.md").write_text(
        f"""# {name.title().replace('-', ' ')}

## Purpose
Describe what this pack does.

## Inputs
- List input requirements

## Outputs
- List outputs produced

## Assumptions
- List key assumptions

## License
{manifest.license}
"""
    )

    if kind == "pack":
        (pack_dir / "gl.yaml").write_text(
            f"""version: 0.1
pipeline:
  name: "{name} pipeline"
inputs:
  params:
    example: "value"
steps:
  - id: step1
    agent: ExampleAgent
    with_:
      param: ${{input:params.example}}
outputs:
  result: ${{ref:step1.data}}
"""
        )

    print(f"Created pack template at {pack_dir}")
    print(f"  - Edit pack.yaml to configure")
    print(f"  - Add pipelines, agents, datasets as needed")
    print(f"  - Write tests in tests/")
    print(f"  - Document in CARD.md")
