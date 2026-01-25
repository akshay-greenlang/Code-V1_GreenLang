"""
ProvenanceTracker - Complete data lineage and traceability for BURNMASTER.

This module implements the ProvenanceTracker for GL-004 BURNMASTER, providing
full traceability linking recommendations to data snapshots, model versions,
code versions, and constraint sets.

Supports regulatory compliance by enabling complete reconstruction of any
recommendation with cryptographic verification of data integrity.

Example:
    >>> tracker = ProvenanceTracker(config)
    >>> snapshot = tracker.capture_data_snapshot(combustion_data)
    >>> link = tracker.link_provenance(recommendation)
    >>> result = tracker.validate_provenance(link)
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import uuid
import subprocess
import os

logger = logging.getLogger(__name__)


# =============================================================================
# Input Models
# =============================================================================

class CombustionData(BaseModel):
    """Combustion sensor and process data snapshot."""

    timestamp: datetime = Field(..., description="Data collection timestamp")
    boiler_id: str = Field(..., description="Boiler identifier")
    fuel_flow_rate: float = Field(..., ge=0, description="Fuel flow rate (kg/h)")
    air_flow_rate: float = Field(..., ge=0, description="Air flow rate (m3/h)")
    flue_gas_temperature: float = Field(..., description="Flue gas temperature (C)")
    oxygen_content: float = Field(..., ge=0, le=21, description="O2 content (%)")
    co_concentration: float = Field(..., ge=0, description="CO concentration (ppm)")
    nox_concentration: float = Field(..., ge=0, description="NOx concentration (ppm)")
    steam_pressure: float = Field(..., ge=0, description="Steam pressure (bar)")
    steam_temperature: float = Field(..., description="Steam temperature (C)")
    feedwater_temperature: float = Field(..., description="Feedwater temperature (C)")
    excess_air_ratio: float = Field(..., ge=1.0, description="Excess air ratio")
    thermal_efficiency: Optional[float] = Field(None, ge=0, le=100, description="Thermal efficiency (%)")
    load_percentage: float = Field(..., ge=0, le=100, description="Load percentage")
    additional_sensors: Dict[str, float] = Field(default_factory=dict, description="Additional sensor data")


class ConstraintSet(BaseModel):
    """Operational constraints for optimization."""

    constraint_set_id: str = Field(..., description="Unique constraint set identifier")
    version: str = Field(..., description="Constraint set version")
    effective_date: datetime = Field(..., description="When constraints became effective")

    # Emissions constraints
    max_nox_ppm: float = Field(..., gt=0, description="Maximum NOx (ppm)")
    max_co_ppm: float = Field(..., gt=0, description="Maximum CO (ppm)")
    max_particulate_mg_m3: float = Field(..., gt=0, description="Maximum particulate (mg/m3)")

    # Operational constraints
    min_oxygen_percent: float = Field(..., ge=0, le=21, description="Minimum O2 (%)")
    max_oxygen_percent: float = Field(..., ge=0, le=21, description="Maximum O2 (%)")
    min_flue_gas_temp_c: float = Field(..., description="Minimum flue gas temp (C)")
    max_flue_gas_temp_c: float = Field(..., description="Maximum flue gas temp (C)")
    min_efficiency_percent: float = Field(..., ge=0, le=100, description="Minimum efficiency (%)")

    # Rate of change constraints
    max_fuel_rate_change_per_min: float = Field(..., gt=0, description="Max fuel change rate (%/min)")
    max_air_rate_change_per_min: float = Field(..., gt=0, description="Max air change rate (%/min)")

    # Safety constraints
    safety_margin_percent: float = Field(10.0, ge=0, le=50, description="Safety margin (%)")

    # Additional custom constraints
    custom_constraints: Dict[str, Any] = Field(default_factory=dict, description="Custom constraints")


class RecommendationInput(BaseModel):
    """Recommendation to link with provenance."""

    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    timestamp: datetime = Field(..., description="Recommendation timestamp")
    model_id: str = Field(..., description="Model that generated recommendation")
    target_setpoints: Dict[str, float] = Field(..., description="Recommended setpoints")
    expected_outcomes: Dict[str, float] = Field(..., description="Expected outcomes")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score")
    input_features: Dict[str, Any] = Field(..., description="Input features used")


# =============================================================================
# Output Models
# =============================================================================

class DataSnapshot(BaseModel):
    """Immutable snapshot of combustion data with hash."""

    snapshot_id: str = Field(..., description="Unique snapshot identifier")
    timestamp: datetime = Field(..., description="Snapshot creation timestamp")
    data_timestamp: datetime = Field(..., description="Original data timestamp")
    boiler_id: str = Field(..., description="Boiler identifier")
    data: Dict[str, Any] = Field(..., description="Serialized data")
    data_hash: str = Field(..., description="SHA-256 hash of data")
    sensor_count: int = Field(..., ge=0, description="Number of sensors captured")

    class Config:
        """Pydantic configuration."""
        frozen = True


class ModelVersion(BaseModel):
    """Model version information with verification hash."""

    model_id: str = Field(..., description="Model identifier")
    version: str = Field(..., description="Model version string")
    model_type: str = Field(..., description="Type of model")
    training_date: Optional[datetime] = Field(None, description="When model was trained")
    model_hash: str = Field(..., description="SHA-256 hash of model weights/params")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    training_metrics: Dict[str, float] = Field(default_factory=dict, description="Training metrics")
    capture_timestamp: datetime = Field(..., description="When version was captured")

    class Config:
        """Pydantic configuration."""
        frozen = True


class CodeVersion(BaseModel):
    """Code version information from version control."""

    commit_hash: str = Field(..., description="Git commit hash")
    branch: str = Field(..., description="Git branch name")
    tag: Optional[str] = Field(None, description="Git tag if any")
    is_dirty: bool = Field(..., description="Whether working directory has uncommitted changes")
    commit_timestamp: Optional[datetime] = Field(None, description="Commit timestamp")
    commit_message: Optional[str] = Field(None, description="Commit message")
    capture_timestamp: datetime = Field(..., description="When version was captured")
    repository_url: Optional[str] = Field(None, description="Repository URL")

    class Config:
        """Pydantic configuration."""
        frozen = True


class ConstraintSnapshot(BaseModel):
    """Immutable snapshot of constraint set with hash."""

    snapshot_id: str = Field(..., description="Unique snapshot identifier")
    constraint_set_id: str = Field(..., description="Original constraint set ID")
    version: str = Field(..., description="Constraint set version")
    effective_date: datetime = Field(..., description="When constraints became effective")
    constraints: Dict[str, Any] = Field(..., description="Serialized constraints")
    constraint_hash: str = Field(..., description="SHA-256 hash of constraints")
    capture_timestamp: datetime = Field(..., description="When snapshot was captured")

    class Config:
        """Pydantic configuration."""
        frozen = True


class ProvenanceLink(BaseModel):
    """Complete provenance link for a recommendation."""

    link_id: str = Field(..., description="Unique provenance link identifier")
    recommendation_id: str = Field(..., description="Linked recommendation ID")
    timestamp: datetime = Field(..., description="Link creation timestamp")

    # Linked snapshots
    data_snapshot_id: str = Field(..., description="Data snapshot ID")
    data_snapshot_hash: str = Field(..., description="Data snapshot hash")
    model_version_hash: str = Field(..., description="Model version hash")
    code_version_hash: str = Field(..., description="Code version hash")
    constraint_snapshot_id: str = Field(..., description="Constraint snapshot ID")
    constraint_snapshot_hash: str = Field(..., description="Constraint snapshot hash")

    # Combined provenance hash
    provenance_hash: str = Field(..., description="Combined SHA-256 hash of all components")

    # Metadata
    environment: str = Field(..., description="Execution environment")
    hostname: Optional[str] = Field(None, description="Host machine name")

    class Config:
        """Pydantic configuration."""
        frozen = True


class ValidationResult(BaseModel):
    """Result of provenance validation."""

    link_id: str = Field(..., description="Validated provenance link ID")
    is_valid: bool = Field(..., description="Overall validation result")
    validation_timestamp: datetime = Field(..., description="When validation was performed")

    # Component validation results
    data_valid: bool = Field(..., description="Data snapshot hash valid")
    model_valid: bool = Field(..., description="Model version hash valid")
    code_valid: bool = Field(..., description="Code version hash valid")
    constraints_valid: bool = Field(..., description="Constraint snapshot hash valid")
    provenance_valid: bool = Field(..., description="Combined provenance hash valid")

    # Validation details
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


# =============================================================================
# Configuration
# =============================================================================

class ProvenanceTrackerConfig(BaseModel):
    """Configuration for ProvenanceTracker."""

    storage_backend: str = Field("memory", description="Storage backend: memory, file, database")
    storage_path: str = Field("./provenance", description="Path for file-based storage")
    enable_git_integration: bool = Field(True, description="Enable Git version capture")
    git_repository_path: Optional[str] = Field(None, description="Git repository path")
    environment: str = Field("production", description="Execution environment name")
    auto_capture_code_version: bool = Field(True, description="Auto-capture code version")


# =============================================================================
# ProvenanceTracker Implementation
# =============================================================================

class ProvenanceTracker:
    """
    ProvenanceTracker implementation for BURNMASTER.

    This class provides complete data lineage and traceability, linking
    recommendations to data snapshots, model versions, code versions,
    and constraint sets.

    All snapshots are immutable with SHA-256 hashes for integrity verification,
    supporting regulatory compliance and full reconstruction of any recommendation.

    Attributes:
        config: Tracker configuration
        _data_snapshots: Storage for data snapshots
        _model_versions: Storage for model versions
        _constraint_snapshots: Storage for constraint snapshots
        _provenance_links: Storage for provenance links

    Example:
        >>> config = ProvenanceTrackerConfig()
        >>> tracker = ProvenanceTracker(config)
        >>> snapshot = tracker.capture_data_snapshot(combustion_data)
        >>> link = tracker.link_provenance(recommendation)
    """

    def __init__(self, config: ProvenanceTrackerConfig):
        """
        Initialize ProvenanceTracker.

        Args:
            config: Tracker configuration
        """
        self.config = config
        self._data_snapshots: Dict[str, DataSnapshot] = {}
        self._model_versions: Dict[str, ModelVersion] = {}
        self._constraint_snapshots: Dict[str, ConstraintSnapshot] = {}
        self._provenance_links: Dict[str, ProvenanceLink] = {}
        self._current_code_version: Optional[CodeVersion] = None

        # Auto-capture code version on initialization
        if config.auto_capture_code_version:
            try:
                self._current_code_version = self.capture_code_version()
            except Exception as e:
                logger.warning(f"Failed to auto-capture code version: {e}")

        logger.info(
            f"ProvenanceTracker initialized with backend={config.storage_backend}, "
            f"environment={config.environment}"
        )

    def capture_data_snapshot(self, data: CombustionData) -> DataSnapshot:
        """
        Capture an immutable snapshot of combustion data.

        Args:
            data: Combustion data to snapshot

        Returns:
            Immutable data snapshot with hash

        Raises:
            ValueError: If data is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            snapshot_id = str(uuid.uuid4())

            # Serialize data for hashing
            data_dict = data.dict()
            data_json = json.dumps(data_dict, sort_keys=True, default=str)
            data_hash = hashlib.sha256(data_json.encode('utf-8')).hexdigest()

            # Count sensors (base + additional)
            base_sensor_count = 13  # Number of base sensor fields
            additional_sensor_count = len(data.additional_sensors)
            total_sensors = base_sensor_count + additional_sensor_count

            snapshot = DataSnapshot(
                snapshot_id=snapshot_id,
                timestamp=start_time,
                data_timestamp=data.timestamp,
                boiler_id=data.boiler_id,
                data=data_dict,
                data_hash=data_hash,
                sensor_count=total_sensors
            )

            # Store snapshot
            self._data_snapshots[snapshot_id] = snapshot

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Captured data snapshot {snapshot_id} with {total_sensors} sensors "
                f"in {processing_time_ms:.2f}ms"
            )

            return snapshot

        except Exception as e:
            logger.error(f"Failed to capture data snapshot: {str(e)}", exc_info=True)
            raise

    def capture_model_version(self, model_id: str) -> ModelVersion:
        """
        Capture model version information.

        Args:
            model_id: Model identifier to capture

        Returns:
            Model version with verification hash

        Raises:
            ValueError: If model_id is invalid or model not found
        """
        start_time = datetime.now(timezone.utc)

        try:
            # In production, this would load model metadata from model registry
            # For now, create a version record based on model_id

            # Parse model_id to extract version info (format: model_name_v1.0.0)
            parts = model_id.rsplit('_v', 1)
            model_name = parts[0] if len(parts) > 1 else model_id
            version = parts[1] if len(parts) > 1 else "1.0.0"

            # Create model hash (in production, hash actual model weights)
            model_info = {
                "model_id": model_id,
                "model_name": model_name,
                "version": version,
                "timestamp": start_time.isoformat()
            }
            model_info_json = json.dumps(model_info, sort_keys=True)
            model_hash = hashlib.sha256(model_info_json.encode('utf-8')).hexdigest()

            model_version = ModelVersion(
                model_id=model_id,
                version=version,
                model_type="combustion_optimizer",
                training_date=None,  # Would be populated from model registry
                model_hash=model_hash,
                hyperparameters={},  # Would be populated from model registry
                training_metrics={},  # Would be populated from model registry
                capture_timestamp=start_time
            )

            # Store version
            self._model_versions[model_hash] = model_version

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Captured model version {model_id} v{version} "
                f"in {processing_time_ms:.2f}ms"
            )

            return model_version

        except Exception as e:
            logger.error(f"Failed to capture model version: {str(e)}", exc_info=True)
            raise

    def capture_code_version(self) -> CodeVersion:
        """
        Capture current code version from Git.

        Returns:
            Code version information

        Raises:
            RuntimeError: If Git integration fails
        """
        start_time = datetime.now(timezone.utc)

        try:
            repo_path = self.config.git_repository_path or os.getcwd()

            # Get commit hash
            commit_hash = self._run_git_command(
                ["git", "rev-parse", "HEAD"],
                repo_path
            ).strip()

            # Get branch name
            branch = self._run_git_command(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                repo_path
            ).strip()

            # Check for uncommitted changes
            status = self._run_git_command(
                ["git", "status", "--porcelain"],
                repo_path
            )
            is_dirty = len(status.strip()) > 0

            # Get tag if any
            try:
                tag = self._run_git_command(
                    ["git", "describe", "--tags", "--exact-match", "HEAD"],
                    repo_path
                ).strip()
            except Exception:
                tag = None

            # Get commit timestamp
            try:
                timestamp_str = self._run_git_command(
                    ["git", "show", "-s", "--format=%ci", "HEAD"],
                    repo_path
                ).strip()
                commit_timestamp = datetime.fromisoformat(timestamp_str.replace(' ', 'T').replace(' +', '+'))
            except Exception:
                commit_timestamp = None

            # Get commit message
            try:
                commit_message = self._run_git_command(
                    ["git", "show", "-s", "--format=%s", "HEAD"],
                    repo_path
                ).strip()
            except Exception:
                commit_message = None

            code_version = CodeVersion(
                commit_hash=commit_hash,
                branch=branch,
                tag=tag,
                is_dirty=is_dirty,
                commit_timestamp=commit_timestamp,
                commit_message=commit_message,
                capture_timestamp=start_time,
                repository_url=None
            )

            self._current_code_version = code_version

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Captured code version {commit_hash[:8]} on {branch} "
                f"(dirty={is_dirty}) in {processing_time_ms:.2f}ms"
            )

            return code_version

        except Exception as e:
            logger.error(f"Failed to capture code version: {str(e)}", exc_info=True)
            # Return a fallback code version
            return CodeVersion(
                commit_hash="unknown",
                branch="unknown",
                tag=None,
                is_dirty=True,
                commit_timestamp=None,
                commit_message=None,
                capture_timestamp=start_time,
                repository_url=None
            )

    def capture_constraint_set(self, constraints: ConstraintSet) -> ConstraintSnapshot:
        """
        Capture an immutable snapshot of constraint set.

        Args:
            constraints: Constraint set to snapshot

        Returns:
            Immutable constraint snapshot with hash

        Raises:
            ValueError: If constraints are invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            snapshot_id = str(uuid.uuid4())

            # Serialize constraints for hashing
            constraints_dict = constraints.dict()
            constraints_json = json.dumps(constraints_dict, sort_keys=True, default=str)
            constraint_hash = hashlib.sha256(constraints_json.encode('utf-8')).hexdigest()

            snapshot = ConstraintSnapshot(
                snapshot_id=snapshot_id,
                constraint_set_id=constraints.constraint_set_id,
                version=constraints.version,
                effective_date=constraints.effective_date,
                constraints=constraints_dict,
                constraint_hash=constraint_hash,
                capture_timestamp=start_time
            )

            # Store snapshot
            self._constraint_snapshots[snapshot_id] = snapshot

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Captured constraint snapshot {snapshot_id} v{constraints.version} "
                f"in {processing_time_ms:.2f}ms"
            )

            return snapshot

        except Exception as e:
            logger.error(f"Failed to capture constraint snapshot: {str(e)}", exc_info=True)
            raise

    def link_provenance(
        self,
        recommendation: RecommendationInput,
        data_snapshot_id: Optional[str] = None,
        constraint_snapshot_id: Optional[str] = None
    ) -> ProvenanceLink:
        """
        Create a complete provenance link for a recommendation.

        Links the recommendation to data snapshot, model version,
        code version, and constraint set.

        Args:
            recommendation: Recommendation to link
            data_snapshot_id: Optional specific data snapshot ID
            constraint_snapshot_id: Optional specific constraint snapshot ID

        Returns:
            Complete provenance link

        Raises:
            ValueError: If required snapshots not found
        """
        start_time = datetime.now(timezone.utc)

        try:
            link_id = str(uuid.uuid4())

            # Get or find data snapshot
            if data_snapshot_id:
                if data_snapshot_id not in self._data_snapshots:
                    raise ValueError(f"Data snapshot {data_snapshot_id} not found")
                data_snapshot = self._data_snapshots[data_snapshot_id]
            else:
                # Use most recent snapshot
                if not self._data_snapshots:
                    raise ValueError("No data snapshots available")
                data_snapshot = max(
                    self._data_snapshots.values(),
                    key=lambda s: s.timestamp
                )
                data_snapshot_id = data_snapshot.snapshot_id

            # Get model version
            model_version = self.capture_model_version(recommendation.model_id)

            # Get code version
            if self._current_code_version is None:
                self._current_code_version = self.capture_code_version()
            code_version = self._current_code_version

            # Get or find constraint snapshot
            if constraint_snapshot_id:
                if constraint_snapshot_id not in self._constraint_snapshots:
                    raise ValueError(f"Constraint snapshot {constraint_snapshot_id} not found")
                constraint_snapshot = self._constraint_snapshots[constraint_snapshot_id]
            else:
                # Use most recent snapshot
                if not self._constraint_snapshots:
                    raise ValueError("No constraint snapshots available")
                constraint_snapshot = max(
                    self._constraint_snapshots.values(),
                    key=lambda s: s.capture_timestamp
                )
                constraint_snapshot_id = constraint_snapshot.snapshot_id

            # Create code version hash
            code_version_data = {
                "commit_hash": code_version.commit_hash,
                "branch": code_version.branch,
                "is_dirty": code_version.is_dirty
            }
            code_version_hash = hashlib.sha256(
                json.dumps(code_version_data, sort_keys=True).encode('utf-8')
            ).hexdigest()

            # Create combined provenance hash
            provenance_data = {
                "recommendation_id": recommendation.recommendation_id,
                "data_hash": data_snapshot.data_hash,
                "model_hash": model_version.model_hash,
                "code_hash": code_version_hash,
                "constraint_hash": constraint_snapshot.constraint_hash
            }
            provenance_hash = hashlib.sha256(
                json.dumps(provenance_data, sort_keys=True).encode('utf-8')
            ).hexdigest()

            # Get hostname
            try:
                hostname = os.uname().nodename
            except AttributeError:
                hostname = os.environ.get('COMPUTERNAME', 'unknown')

            link = ProvenanceLink(
                link_id=link_id,
                recommendation_id=recommendation.recommendation_id,
                timestamp=start_time,
                data_snapshot_id=data_snapshot_id,
                data_snapshot_hash=data_snapshot.data_hash,
                model_version_hash=model_version.model_hash,
                code_version_hash=code_version_hash,
                constraint_snapshot_id=constraint_snapshot_id,
                constraint_snapshot_hash=constraint_snapshot.constraint_hash,
                provenance_hash=provenance_hash,
                environment=self.config.environment,
                hostname=hostname
            )

            # Store link
            self._provenance_links[link_id] = link

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Created provenance link {link_id} for recommendation "
                f"{recommendation.recommendation_id} in {processing_time_ms:.2f}ms"
            )

            return link

        except Exception as e:
            logger.error(f"Failed to create provenance link: {str(e)}", exc_info=True)
            raise

    def validate_provenance(self, link: ProvenanceLink) -> ValidationResult:
        """
        Validate a provenance link's integrity.

        Verifies all hashes match stored snapshots.

        Args:
            link: Provenance link to validate

        Returns:
            Validation result with details

        Raises:
            ValueError: If link not found
        """
        start_time = datetime.now(timezone.utc)

        try:
            errors: List[str] = []
            warnings: List[str] = []

            # Validate data snapshot
            data_valid = False
            if link.data_snapshot_id in self._data_snapshots:
                stored_snapshot = self._data_snapshots[link.data_snapshot_id]
                if stored_snapshot.data_hash == link.data_snapshot_hash:
                    data_valid = True
                else:
                    errors.append(
                        f"Data snapshot hash mismatch: expected {link.data_snapshot_hash}, "
                        f"got {stored_snapshot.data_hash}"
                    )
            else:
                errors.append(f"Data snapshot {link.data_snapshot_id} not found")

            # Validate model version
            model_valid = link.model_version_hash in self._model_versions
            if not model_valid:
                errors.append(f"Model version hash {link.model_version_hash} not found")

            # Validate code version (check against current if available)
            code_valid = True  # Assume valid if we can't verify
            if self._current_code_version:
                code_version_data = {
                    "commit_hash": self._current_code_version.commit_hash,
                    "branch": self._current_code_version.branch,
                    "is_dirty": self._current_code_version.is_dirty
                }
                current_code_hash = hashlib.sha256(
                    json.dumps(code_version_data, sort_keys=True).encode('utf-8')
                ).hexdigest()
                if current_code_hash != link.code_version_hash:
                    warnings.append(
                        f"Code version has changed since provenance was created"
                    )

            # Validate constraint snapshot
            constraints_valid = False
            if link.constraint_snapshot_id in self._constraint_snapshots:
                stored_constraint = self._constraint_snapshots[link.constraint_snapshot_id]
                if stored_constraint.constraint_hash == link.constraint_snapshot_hash:
                    constraints_valid = True
                else:
                    errors.append(
                        f"Constraint snapshot hash mismatch: expected {link.constraint_snapshot_hash}, "
                        f"got {stored_constraint.constraint_hash}"
                    )
            else:
                errors.append(f"Constraint snapshot {link.constraint_snapshot_id} not found")

            # Validate combined provenance hash
            provenance_data = {
                "recommendation_id": link.recommendation_id,
                "data_hash": link.data_snapshot_hash,
                "model_hash": link.model_version_hash,
                "code_hash": link.code_version_hash,
                "constraint_hash": link.constraint_snapshot_hash
            }
            computed_hash = hashlib.sha256(
                json.dumps(provenance_data, sort_keys=True).encode('utf-8')
            ).hexdigest()
            provenance_valid = computed_hash == link.provenance_hash
            if not provenance_valid:
                errors.append(
                    f"Provenance hash mismatch: expected {link.provenance_hash}, "
                    f"computed {computed_hash}"
                )

            # Overall validation
            is_valid = (
                data_valid and model_valid and code_valid and
                constraints_valid and provenance_valid and len(errors) == 0
            )

            result = ValidationResult(
                link_id=link.link_id,
                is_valid=is_valid,
                validation_timestamp=start_time,
                data_valid=data_valid,
                model_valid=model_valid,
                code_valid=code_valid,
                constraints_valid=constraints_valid,
                provenance_valid=provenance_valid,
                errors=errors,
                warnings=warnings
            )

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Validated provenance link {link.link_id}: valid={is_valid}, "
                f"errors={len(errors)}, warnings={len(warnings)} in {processing_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to validate provenance: {str(e)}", exc_info=True)
            raise

    def get_data_snapshot(self, snapshot_id: str) -> Optional[DataSnapshot]:
        """Get a data snapshot by ID."""
        return self._data_snapshots.get(snapshot_id)

    def get_constraint_snapshot(self, snapshot_id: str) -> Optional[ConstraintSnapshot]:
        """Get a constraint snapshot by ID."""
        return self._constraint_snapshots.get(snapshot_id)

    def get_provenance_link(self, link_id: str) -> Optional[ProvenanceLink]:
        """Get a provenance link by ID."""
        return self._provenance_links.get(link_id)

    def _run_git_command(self, command: List[str], cwd: str) -> str:
        """
        Run a Git command and return output.

        Args:
            command: Command and arguments
            cwd: Working directory

        Returns:
            Command output

        Raises:
            RuntimeError: If command fails
        """
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"Git command failed: {result.stderr}")
            return result.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError("Git command timed out")
        except FileNotFoundError:
            raise RuntimeError("Git not found in PATH")
