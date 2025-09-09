"""
Execution context and artifacts
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel


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
    
    def __init__(self, 
                 artifacts_dir: Optional[Path] = None,
                 profile: str = "dev",
                 backend: str = "local",
                 metadata: Optional[Dict[str, Any]] = None):
        self.artifacts_dir = artifacts_dir or Path("out")
        self.profile = profile
        self.backend = backend
        self.metadata = metadata or {}
        self.artifacts: List[Artifact] = []
        self.start_time = datetime.utcnow()
        
        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def add_artifact(self, name: str, path: Path, type: str = "file", **metadata) -> Artifact:
        """Add an artifact to the context"""
        artifact = Artifact(
            name=name,
            path=path,
            type=type,
            metadata=metadata
        )
        self.artifacts.append(artifact)
        return artifact
    
    def get_artifact(self, name: str) -> Optional[Artifact]:
        """Get an artifact by name"""
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "artifacts_dir": str(self.artifacts_dir),
            "profile": self.profile,
            "backend": self.backend,
            "metadata": self.metadata,
            "artifacts": [a.model_dump() if hasattr(a, 'model_dump') else a.dict() for a in self.artifacts],
            "start_time": self.start_time.isoformat(),
            "duration": (datetime.utcnow() - self.start_time).total_seconds()
        }