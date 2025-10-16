"""
GreenLang Provenance - Records Module
Provenance record structures and serialization.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PROVENANCE RECORD
# ============================================================================

@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for audit trails.

    This dataclass captures all information needed to audit and reproduce
    an execution or report generation.

    Attributes:
        record_id: Unique record identifier
        generated_at: Timestamp (ISO 8601)
        input_file_hash: SHA256 hash of input file (if applicable)
        environment: Execution environment details
        dependencies: Package versions
        configuration: Configuration snapshot
        agent_execution: Agent execution details
        data_lineage: Data transformation lineage
        validation_results: Validation outcomes
        metadata: Additional metadata
    """
    record_id: str
    generated_at: str
    environment: Dict[str, Any]
    dependencies: Dict[str, str]
    configuration: Dict[str, Any]
    agent_execution: List[Dict[str, Any]] = field(default_factory=list)
    data_lineage: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    input_file_hash: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str):
        """
        Save provenance record to file.

        Args:
            path: File path to save to
        """
        with open(path, 'w') as f:
            f.write(self.to_json())

        logger.info(f"Provenance record saved to {path}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        """
        Create from dictionary.

        Args:
            data: Dictionary with provenance data

        Returns:
            ProvenanceRecord instance
        """
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "ProvenanceRecord":
        """
        Create from JSON string.

        Args:
            json_str: JSON string

        Returns:
            ProvenanceRecord instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str) -> "ProvenanceRecord":
        """
        Load provenance record from file.

        Args:
            path: File path to load from

        Returns:
            ProvenanceRecord instance
        """
        with open(path, 'r') as f:
            return cls.from_json(f.read())


# ============================================================================
# PROVENANCE CONTEXT (For Runtime Tracking)
# ============================================================================

class ProvenanceContext:
    """
    Runtime provenance tracking context.

    Used to collect provenance information during execution.

    Example:
        >>> ctx = ProvenanceContext("my_pipeline")
        >>> ctx.record_input("data.csv", {"rows": 1000})
        >>> ctx.record_agent_execution("ValidatorAgent", {...})
        >>> ctx.finalize()
        >>> provenance = ctx.to_record()
    """

    def __init__(self, name: str = "default", record_id: Optional[str] = None):
        """
        Initialize provenance context.

        Args:
            name: Context name
            record_id: Optional record ID (generated if not provided)
        """
        self.name = name
        self.record_id = record_id or self._generate_record_id()
        self.started_at = datetime.now(timezone.utc)

        # Tracking data
        self.inputs: List[Dict[str, Any]] = []
        self.outputs: Dict[str, Any] = {}
        self.configuration: Dict[str, Any] = {}
        self.agent_executions: List[Dict[str, Any]] = []
        self.data_lineage: List[Dict[str, Any]] = []
        self.validation_results: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

        # Environment snapshot (captured on init)
        from .environment import get_environment_info, get_dependency_versions
        self.environment = get_environment_info()
        self.dependencies = get_dependency_versions()

    def _generate_record_id(self) -> str:
        """Generate unique record ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{self.name}-{timestamp}"

    def record_input(self, source: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Record an input source.

        Args:
            source: Input source (file path, URL, etc.)
            metadata: Optional metadata about input
        """
        from .hashing import hash_file

        input_record = {
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }

        # Hash file if it's a path
        try:
            if Path(source).exists():
                input_record["file_hash"] = hash_file(source)
        except Exception:
            pass

        self.inputs.append(input_record)

    def record_output(self, destination: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Record an output.

        Args:
            destination: Output destination
            metadata: Optional metadata about output
        """
        self.outputs[destination] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }

    def record_agent_execution(
        self,
        agent_name: str,
        start_time: str,
        end_time: str,
        duration_seconds: float,
        input_records: int = 0,
        output_records: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record agent execution details.

        Args:
            agent_name: Name of the agent
            start_time: Start timestamp
            end_time: End timestamp
            duration_seconds: Execution duration
            input_records: Number of input records processed
            output_records: Number of output records produced
            metadata: Optional additional metadata
        """
        execution = {
            "agent_name": agent_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
            "input_records": input_records,
            "output_records": output_records,
            "metadata": metadata or {}
        }

        self.agent_executions.append(execution)

        # Also add to data lineage
        self.data_lineage.append({
            "step": len(self.data_lineage),
            "stage": agent_name,
            "description": f"{agent_name} processed {input_records} records → {output_records} records",
            "input_records": input_records,
            "output_records": output_records,
            "timestamp": end_time
        })

    def record_validation(self, validation_results: Dict[str, Any]):
        """
        Record validation results.

        Args:
            validation_results: Validation outcome
        """
        self.validation_results = validation_results

    def set_configuration(self, config: Dict[str, Any]):
        """
        Set configuration snapshot.

        Args:
            config: Configuration dictionary
        """
        self.configuration = config

    def add_metadata(self, key: str, value: Any):
        """
        Add metadata key-value pair.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def to_record(self) -> ProvenanceRecord:
        """
        Convert context to ProvenanceRecord.

        Returns:
            ProvenanceRecord with all collected data
        """
        # Get input file hash if there's exactly one input file
        input_file_hash = None
        if len(self.inputs) == 1 and "file_hash" in self.inputs[0]:
            input_file_hash = self.inputs[0]["file_hash"]

        return ProvenanceRecord(
            record_id=self.record_id,
            generated_at=self.started_at.isoformat(),
            environment=self.environment,
            dependencies=self.dependencies,
            configuration=self.configuration,
            agent_execution=self.agent_executions,
            data_lineage=self.data_lineage,
            validation_results=self.validation_results,
            input_file_hash=input_file_hash,
            metadata=self.metadata
        )

    def finalize(self, output_path: Optional[str] = None) -> ProvenanceRecord:
        """
        Finalize and save provenance record.

        Args:
            output_path: Optional path to save record

        Returns:
            Final ProvenanceRecord
        """
        record = self.to_record()

        if output_path:
            record.save(output_path)

        return record
