"""
Pipeline execution and management
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class Pipeline(BaseModel):
    """Pipeline runner and manager"""

    name: str
    version: str = "1.0"
    description: Optional[str] = None
    inputs: Dict[str, Any] = {}
    steps: List[Dict[str, Any]] = []
    outputs: Dict[str, Any] = {}

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        """Load pipeline from YAML file"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save pipeline to YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump() if hasattr(self, "model_dump") else self.dict()

    def load_inputs_file(self, path: str) -> None:
        """Load inputs from file"""
        if Path(path).suffix == ".json":
            import json

            with open(path) as f:
                self.inputs.update(json.load(f))
        elif Path(path).suffix in [".yaml", ".yml"]:
            with open(path) as f:
                self.inputs.update(yaml.safe_load(f))

    def validate(self) -> List[str]:
        """Validate pipeline structure"""
        errors = []

        if not self.name:
            errors.append("Pipeline name is required")

        if not self.steps:
            errors.append("Pipeline must have at least one step")

        for i, step in enumerate(self.steps):
            if "name" not in step:
                errors.append(f"Step {i} missing name")
            if "agent" not in step and "pipeline" not in step:
                errors.append(f"Step {i} must specify agent or pipeline")

        return errors
