"""
GreenLang Provenance - Tracker Module
Automatic provenance tracking for sustainability operations.

Provides high-level provenance tracking with automatic lineage capture,
chain-of-custody tracking, and integrity verification.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
from functools import wraps

from .records import ProvenanceRecord, ProvenanceContext
from .hashing import hash_file, hash_data
from .environment import get_environment_info, get_dependency_versions, get_system_info
from .validation import validate_provenance, verify_integrity

logger = logging.getLogger(__name__)


class ProvenanceTracker:
    """
    High-level provenance tracker for automatic lineage tracking.

    Features:
    - Automatic lineage tracking
    - Chain-of-custody tracking
    - SHA-256 hashing for integrity
    - Context managers for automatic tracking
    - Decorator-based tracking
    - Multi-level tracking (operation, pipeline, system)

    Example:
        >>> tracker = ProvenanceTracker()
        >>>
        >>> # Track an operation
        >>> with tracker.track_operation("data_intake") as ctx:
        >>>     ctx.record_input("data.csv")
        >>>     # ... process data ...
        >>>     ctx.record_output("cleaned_data.parquet")
        >>>
        >>> # Get provenance record
        >>> record = tracker.get_record()
        >>> record.save("provenance.json")
    """

    def __init__(
        self,
        name: str = "default",
        auto_capture_env: bool = True,
        auto_hash_files: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize provenance tracker.

        Args:
            name: Tracker name/identifier
            auto_capture_env: Automatically capture environment info
            auto_hash_files: Automatically hash input/output files
            config: Optional configuration
        """
        self.name = name
        self.auto_capture_env = auto_capture_env
        self.auto_hash_files = auto_hash_files
        self.config = config or {}

        # Initialize tracking context
        self.context = ProvenanceContext(name=name)

        # Chain of custody tracking
        self.chain_of_custody: List[Dict[str, Any]] = []

        # Operation stack for nested tracking
        self._operation_stack: List[Dict[str, Any]] = []

        logger.info(f"Initialized ProvenanceTracker: {name}")

    @contextmanager
    def track_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracking an operation.

        Args:
            operation_name: Name of the operation
            metadata: Optional operation metadata

        Yields:
            ProvenanceContext for recording details

        Example:
            >>> with tracker.track_operation("data_validation") as ctx:
            >>>     ctx.record_input("input.csv")
            >>>     # ... validate data ...
            >>>     ctx.record_output("valid_data.csv")
        """
        start_time = datetime.now(timezone.utc)
        operation = {
            "name": operation_name,
            "start_time": start_time.isoformat(),
            "metadata": metadata or {}
        }

        self._operation_stack.append(operation)

        try:
            yield self.context

            # Mark as successful
            operation["status"] = "success"

        except Exception as e:
            # Mark as failed
            operation["status"] = "failed"
            operation["error"] = str(e)
            logger.error(f"Operation {operation_name} failed: {e}", exc_info=True)
            raise

        finally:
            end_time = datetime.now(timezone.utc)
            operation["end_time"] = end_time.isoformat()
            operation["duration_seconds"] = (end_time - start_time).total_seconds()

            # Pop from stack
            self._operation_stack.pop()

            # Add to chain of custody
            self.chain_of_custody.append(operation)

    def track_file_input(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Track a file input with automatic hashing.

        Args:
            file_path: Path to input file
            metadata: Optional file metadata

        Returns:
            File tracking information including hash
        """
        file_info = {
            "path": file_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }

        # Get file stats
        path_obj = Path(file_path)
        if path_obj.exists():
            file_info["size_bytes"] = path_obj.stat().st_size
            file_info["modified_time"] = datetime.fromtimestamp(
                path_obj.stat().st_mtime,
                tz=timezone.utc
            ).isoformat()

            # Hash file if enabled
            if self.auto_hash_files:
                file_info["hash"] = hash_file(file_path)

        # Record in context
        self.context.record_input(file_path, file_info)

        return file_info

    def track_file_output(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Track a file output with automatic hashing.

        Args:
            file_path: Path to output file
            metadata: Optional file metadata

        Returns:
            File tracking information including hash
        """
        file_info = {
            "path": file_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }

        # Get file stats
        path_obj = Path(file_path)
        if path_obj.exists():
            file_info["size_bytes"] = path_obj.stat().st_size

            # Hash file if enabled
            if self.auto_hash_files:
                file_info["hash"] = hash_file(file_path)

        # Record in context
        self.context.record_output(file_path, file_info)

        return file_info

    def track_data_transformation(
        self,
        source: str,
        destination: str,
        transformation: str,
        input_records: int = 0,
        output_records: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track a data transformation step in lineage.

        Args:
            source: Source data identifier
            destination: Destination data identifier
            transformation: Description of transformation
            input_records: Number of input records
            output_records: Number of output records
            metadata: Optional transformation metadata
        """
        lineage_entry = {
            "source": source,
            "destination": destination,
            "transformation": transformation,
            "input_records": input_records,
            "output_records": output_records,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }

        self.context.data_lineage.append(lineage_entry)
        logger.debug(f"Tracked transformation: {source} -> {destination}")

    def track_agent_execution(
        self,
        agent_name: str,
        agent_class: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        duration_seconds: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track agent execution with inputs and outputs.

        Args:
            agent_name: Name of the agent
            agent_class: Agent class name
            inputs: Input parameters/data
            outputs: Output results/data
            duration_seconds: Execution duration
            metadata: Optional agent metadata
        """
        execution = {
            "agent_name": agent_name,
            "agent_class": agent_class,
            "inputs": inputs,
            "outputs": outputs,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }

        self.context.agent_executions.append(execution)
        logger.debug(f"Tracked agent execution: {agent_name}")

    def add_custody_transfer(
        self,
        from_entity: str,
        to_entity: str,
        asset: str,
        transfer_type: str = "data",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a chain-of-custody transfer record.

        Args:
            from_entity: Source entity
            to_entity: Destination entity
            asset: Asset being transferred
            transfer_type: Type of transfer (data, report, artifact)
            metadata: Optional transfer metadata
        """
        transfer = {
            "from": from_entity,
            "to": to_entity,
            "asset": asset,
            "type": transfer_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }

        self.chain_of_custody.append(transfer)
        logger.debug(f"Tracked custody transfer: {asset} from {from_entity} to {to_entity}")

    def set_configuration(self, config: Dict[str, Any]):
        """
        Set configuration snapshot.

        Args:
            config: Configuration dictionary
        """
        self.context.set_configuration(config)

    def add_metadata(self, key: str, value: Any):
        """
        Add metadata to provenance record.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.context.add_metadata(key, value)

    def get_record(self) -> ProvenanceRecord:
        """
        Get current provenance record.

        Returns:
            ProvenanceRecord with all tracked information
        """
        record = self.context.to_record()

        # Add chain of custody
        record.metadata["chain_of_custody"] = self.chain_of_custody

        # Add operation history
        record.metadata["operations"] = self.chain_of_custody

        return record

    def save_record(self, output_path: str) -> str:
        """
        Save provenance record to file.

        Args:
            output_path: Path to save record

        Returns:
            Path to saved record
        """
        record = self.get_record()
        record.save(output_path)
        logger.info(f"Saved provenance record to {output_path}")
        return output_path

    def verify_integrity(self, file_path: str) -> bool:
        """
        Verify integrity of a tracked file.

        Args:
            file_path: Path to file to verify

        Returns:
            True if integrity verified, False otherwise
        """
        # Find file in inputs or outputs
        record = self.get_record()

        # Check inputs
        for input_record in self.context.inputs:
            if input_record.get("source") == file_path:
                stored_hash = input_record.get("file_hash")
                if stored_hash:
                    current_hash = hash_file(file_path)
                    return stored_hash == current_hash

        # Check outputs
        for output_path, output_data in self.context.outputs.items():
            if output_path == file_path:
                stored_hash = output_data.get("metadata", {}).get("hash")
                if stored_hash:
                    current_hash = hash_file(file_path)
                    return stored_hash == current_hash

        logger.warning(f"No provenance record found for {file_path}")
        return False

    def generate_audit_trail(self) -> Dict[str, Any]:
        """
        Generate comprehensive audit trail.

        Returns:
            Audit trail with all provenance information
        """
        record = self.get_record()

        audit_trail = {
            "record_id": record.record_id,
            "generated_at": record.generated_at,
            "environment": record.environment,
            "dependencies": record.dependencies,
            "configuration": record.configuration,
            "chain_of_custody": self.chain_of_custody,
            "operations": self.chain_of_custody,
            "agent_executions": record.agent_execution,
            "data_lineage": record.data_lineage,
            "validation_results": record.validation_results,
            "metadata": record.metadata,
        }

        return audit_trail

    def reset(self):
        """Reset tracker to initial state."""
        self.context = ProvenanceContext(name=self.name)
        self.chain_of_custody = []
        self._operation_stack = []
        logger.info(f"Reset ProvenanceTracker: {self.name}")


# ============================================================================
# DECORATOR FOR AUTOMATIC TRACKING
# ============================================================================

def track_with_provenance(
    tracker: Optional[ProvenanceTracker] = None,
    operation_name: Optional[str] = None
):
    """
    Decorator for automatic provenance tracking.

    Args:
        tracker: ProvenanceTracker instance (creates new if None)
        operation_name: Operation name (uses function name if None)

    Example:
        >>> tracker = ProvenanceTracker()
        >>>
        >>> @track_with_provenance(tracker, "data_processing")
        >>> def process_data(input_file, output_file):
        >>>     # ... process data ...
        >>>     return result
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal tracker, operation_name

            # Create tracker if not provided
            if tracker is None:
                tracker = ProvenanceTracker(name=func.__name__)

            # Use function name if operation name not provided
            if operation_name is None:
                operation_name = func.__name__

            # Track operation
            with tracker.track_operation(operation_name):
                result = func(*args, **kwargs)

            return result

        return wrapper
    return decorator


# ============================================================================
# GLOBAL TRACKER (Singleton Pattern)
# ============================================================================

_global_tracker: Optional[ProvenanceTracker] = None


def get_global_tracker(name: str = "global") -> ProvenanceTracker:
    """
    Get or create global provenance tracker.

    Args:
        name: Tracker name

    Returns:
        Global ProvenanceTracker instance
    """
    global _global_tracker

    if _global_tracker is None:
        _global_tracker = ProvenanceTracker(name=name)

    return _global_tracker


def reset_global_tracker():
    """Reset global provenance tracker."""
    global _global_tracker

    if _global_tracker is not None:
        _global_tracker.reset()
    else:
        _global_tracker = None
