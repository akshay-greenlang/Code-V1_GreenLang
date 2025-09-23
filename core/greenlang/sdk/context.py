"""
Execution context and artifacts
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from ..sdk.base import Result


class Artifact(BaseModel):
    """Represents an output artifact"""

    name: str
    path: Path
    type: str
    metadata: Dict[str, Any] = {}
    created_at: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class Context:
    """Execution context for pipelines and agents"""

    def __init__(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        artifacts_dir: Optional[Path] = None,
        profile: str = "dev",
        backend: str = "local",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.inputs = inputs or {}
        self.data = inputs or {}  # Alias for compatibility
        self.artifacts_dir = artifacts_dir or Path("out")
        self.profile = profile
        self.backend = backend
        self.metadata = metadata or {}
        self.artifacts: Dict[str, Artifact] = {}  # Changed to dict for easier lookup
        self.start_time = datetime.utcnow()
        self.steps = {}  # Store step results

        # Add timestamp to metadata
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.utcnow().isoformat()

        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def add_artifact(
        self, name: str, path: Path, type: str = "file", **metadata
    ) -> Artifact:
        """Add an artifact to the context"""
        artifact = Artifact(name=name, path=path, type=type, metadata=metadata)
        self.artifacts[name] = artifact
        return artifact

    def get_artifact(self, name: str) -> Optional[Artifact]:
        """Get an artifact by name"""
        return self.artifacts.get(name)

    def list_artifacts(self) -> List[str]:
        """List all artifact names"""
        return list(self.artifacts.keys())

    def remove_artifact(self, name: str) -> bool:
        """Remove an artifact by name"""
        if name in self.artifacts:
            del self.artifacts[name]
            return True
        return False

    def save_artifact(
        self, name: str, content: Any, type: str = "json", **metadata
    ) -> Artifact:
        """Save content as an artifact file"""
        import json
        import yaml

        # Determine file extension based on type
        ext_map = {"json": ".json", "yaml": ".yaml", "text": ".txt", "csv": ".csv"}
        ext = ext_map.get(type, ".dat")

        # Create artifact file path
        artifact_path = self.artifacts_dir / f"{name}{ext}"

        # Save content based on type
        if type == "json":
            with open(artifact_path, "w") as f:
                json.dump(content, f, indent=2)
        elif type == "yaml":
            with open(artifact_path, "w") as f:
                yaml.dump(content, f)
        elif type == "text":
            with open(artifact_path, "w") as f:
                f.write(str(content))
        else:
            # Default to json
            with open(artifact_path, "w") as f:
                json.dump(content, f, indent=2)

        # Add to artifacts
        return self.add_artifact(name, artifact_path, type=type, **metadata)

    def add_step_result(self, name: str, result: Result):
        """Add a step result to the context"""
        self.steps[name] = {
            "outputs": result.data if hasattr(result, "data") else result,
            "success": result.success if hasattr(result, "success") else True,
            "metadata": result.metadata if hasattr(result, "metadata") else {},
        }
        # Update data with step outputs for next steps to access
        if hasattr(result, "data") and result.data:
            self.data.update({name: result.data})

    def get_step_output(self, step_name: str) -> Optional[Any]:
        """Get output from a previous step"""
        if step_name in self.steps:
            return self.steps[step_name].get("outputs")
        return None

    def get_all_step_outputs(self) -> Dict[str, Any]:
        """Get outputs from all previous steps"""
        return {
            name: step["outputs"]
            for name, step in self.steps.items()
            if "outputs" in step
        }

    def to_result(self) -> Result:
        """Convert context to a Result object"""
        all_success = all(step.get("success", False) for step in self.steps.values())
        return Result(
            success=all_success,
            data=self.steps,
            metadata={
                "inputs": self.inputs,
                "profile": self.profile,
                "backend": self.backend,
                "duration": (datetime.utcnow() - self.start_time).total_seconds(),
                "artifacts": [
                    a.model_dump() if hasattr(a, "model_dump") else a.dict()
                    for a in self.artifacts.values()
                ],
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "inputs": self.inputs,
            "artifacts_dir": str(self.artifacts_dir),
            "profile": self.profile,
            "backend": self.backend,
            "metadata": self.metadata,
            "artifacts": [
                a.model_dump() if hasattr(a, "model_dump") else a.dict()
                for a in self.artifacts.values()
            ],
            "start_time": self.start_time.isoformat(),
            "duration": (datetime.utcnow() - self.start_time).total_seconds(),
            "steps": self.steps,
        }
