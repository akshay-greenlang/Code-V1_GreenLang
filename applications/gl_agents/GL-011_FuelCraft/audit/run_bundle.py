# -*- coding: utf-8 -*-
"""
RunBundle - Immutable run bundle management for GL-011 FuelCraft.

This module implements run bundle creation and management with content-addressed
storage using SHA-256 hashing. Bundles are immutable and support 7-year retention
for regulatory compliance with identical replay across environments.

Key Features:
- Content-addressed storage with SHA-256
- Immutable bundles (append-only)
- Complete input/output capture
- Configuration versioning
- Replay validation support
- 7-year retention compliance

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, BinaryIO
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import hashlib
import json
import gzip
import uuid
import logging

logger = logging.getLogger(__name__)


class BundleStatus(str, Enum):
    """Status of a run bundle."""
    BUILDING = "building"
    SEALED = "sealed"
    ARCHIVED = "archived"
    CORRUPTED = "corrupted"


class ComponentType(str, Enum):
    """Type of component in a bundle."""
    INPUT_SNAPSHOT = "input_snapshot"
    OUTPUT_DATA = "output_data"
    CONVERSION_LOG = "conversion_log"
    MASTER_DATA = "master_data"
    MODEL_VERSION = "model_version"
    SOLVER_CONFIG = "solver_config"
    EXECUTION_LOG = "execution_log"
    METADATA = "metadata"


class BundleComponent(BaseModel):
    """Individual component within a run bundle."""
    component_id: str = Field(...)
    component_type: ComponentType = Field(...)
    filename: str = Field(...)
    content_hash: str = Field(...)
    size_bytes: int = Field(..., ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    compressed: bool = Field(False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BundleManifest(BaseModel):
    """Manifest describing all components in a bundle."""
    bundle_id: str = Field(...)
    run_id: str = Field(...)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sealed_at: Optional[datetime] = Field(None)
    status: BundleStatus = Field(BundleStatus.BUILDING)

    # Provenance
    agent_id: str = Field("GL-011")
    agent_version: str = Field(...)
    environment: str = Field(...)

    # Components
    components: List[BundleComponent] = Field(default_factory=list)

    # Hashes
    manifest_hash: Optional[str] = Field(None)
    bundle_hash: Optional[str] = Field(None)

    # Retention
    retention_years: int = Field(7)
    retention_expires: Optional[datetime] = Field(None)


class ReplayValidationResult(BaseModel):
    """Result of bundle replay validation."""
    bundle_id: str = Field(...)
    validation_id: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_identical: bool = Field(...)
    components_validated: int = Field(0)
    components_matched: int = Field(0)
    mismatches: List[Dict[str, Any]] = Field(default_factory=list)
    original_hash: str = Field(...)
    replay_hash: str = Field(...)


class RunBundleBuilder:
    """
    Builder for creating immutable run bundles.

    Creates content-addressed bundles with SHA-256 hashing for
    complete reproducibility and audit trail.

    Example:
        >>> builder = RunBundleBuilder(run_id="RUN-001", agent_version="1.0.0")
        >>> builder.add_input_snapshot("inventory", inventory_data)
        >>> builder.add_solver_config(solver_config)
        >>> builder.add_output(optimization_result)
        >>> bundle = builder.seal()
    """

    def __init__(
        self,
        run_id: str,
        agent_version: str,
        environment: str = "production",
        storage_path: Optional[str] = None,
        retention_years: int = 7
    ):
        """Initialize bundle builder."""
        self._bundle_id = f"BUNDLE-{uuid.uuid4().hex[:12].upper()}"
        self._run_id = run_id
        self._storage_path = Path(storage_path) if storage_path else None

        self._manifest = BundleManifest(
            bundle_id=self._bundle_id,
            run_id=run_id,
            agent_version=agent_version,
            environment=environment,
            retention_years=retention_years
        )

        self._components_data: Dict[str, bytes] = {}
        self._sealed = False

        logger.info(f"RunBundleBuilder initialized: {self._bundle_id} for run {run_id}")

    @property
    def bundle_id(self) -> str:
        """Get bundle ID."""
        return self._bundle_id

    @property
    def is_sealed(self) -> bool:
        """Check if bundle is sealed."""
        return self._sealed

    def add_input_snapshot(
        self,
        name: str,
        data: Any,
        version: Optional[str] = None
    ) -> str:
        """
        Add an input snapshot to the bundle.

        Args:
            name: Snapshot name (e.g., "inventory", "prices")
            data: Input data (will be JSON serialized)
            version: Optional data version

        Returns:
            Component ID
        """
        return self._add_component(
            component_type=ComponentType.INPUT_SNAPSHOT,
            name=name,
            data=data,
            metadata={"version": version} if version else {}
        )

    def add_output(
        self,
        name: str,
        data: Any,
        output_type: str = "result"
    ) -> str:
        """
        Add output data to the bundle.

        Args:
            name: Output name
            data: Output data
            output_type: Type of output (result, recommendation, report)

        Returns:
            Component ID
        """
        return self._add_component(
            component_type=ComponentType.OUTPUT_DATA,
            name=name,
            data=data,
            metadata={"output_type": output_type}
        )

    def add_conversion_log(
        self,
        conversions: List[Dict[str, Any]]
    ) -> str:
        """
        Add unit conversion log.

        Args:
            conversions: List of conversion records

        Returns:
            Component ID
        """
        return self._add_component(
            component_type=ComponentType.CONVERSION_LOG,
            name="unit_conversions",
            data={"conversions": conversions, "count": len(conversions)},
            metadata={"conversion_count": len(conversions)}
        )

    def add_master_data(
        self,
        name: str,
        data: Any,
        version: str,
        effective_date: datetime
    ) -> str:
        """
        Add master data version record.

        Args:
            name: Master data name (e.g., "emission_factors", "fuel_specs")
            data: Master data content
            version: Data version
            effective_date: When this version became effective

        Returns:
            Component ID
        """
        return self._add_component(
            component_type=ComponentType.MASTER_DATA,
            name=name,
            data=data,
            metadata={
                "version": version,
                "effective_date": effective_date.isoformat()
            }
        )

    def add_model_version(
        self,
        model_name: str,
        version: str,
        model_hash: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add model version record.

        Args:
            model_name: Name of the model
            version: Model version
            model_hash: SHA-256 hash of model artifact
            parameters: Model parameters used

        Returns:
            Component ID
        """
        return self._add_component(
            component_type=ComponentType.MODEL_VERSION,
            name=model_name,
            data={
                "model_name": model_name,
                "version": version,
                "model_hash": model_hash,
                "parameters": parameters or {}
            },
            metadata={"model_hash": model_hash}
        )

    def add_solver_config(
        self,
        solver_name: str,
        config: Dict[str, Any],
        tolerances: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Add solver configuration.

        Args:
            solver_name: Name of solver
            config: Solver configuration
            tolerances: Solver tolerances

        Returns:
            Component ID
        """
        return self._add_component(
            component_type=ComponentType.SOLVER_CONFIG,
            name=solver_name,
            data={
                "solver_name": solver_name,
                "config": config,
                "tolerances": tolerances or {}
            },
            metadata={"solver": solver_name}
        )

    def add_execution_log(
        self,
        logs: List[Dict[str, Any]]
    ) -> str:
        """
        Add execution logs.

        Args:
            logs: List of log entries

        Returns:
            Component ID
        """
        return self._add_component(
            component_type=ComponentType.EXECUTION_LOG,
            name="execution_log",
            data={"logs": logs, "count": len(logs)},
            metadata={"log_count": len(logs)}
        )

    def seal(self) -> BundleManifest:
        """
        Seal the bundle making it immutable.

        Calculates final hashes and marks bundle as sealed.

        Returns:
            Final bundle manifest

        Raises:
            ValueError: If bundle has no components
        """
        if self._sealed:
            raise ValueError("Bundle is already sealed")

        if not self._manifest.components:
            raise ValueError("Cannot seal empty bundle")

        # Calculate manifest hash
        manifest_data = self._manifest.model_dump(exclude={"manifest_hash", "bundle_hash", "sealed_at"})
        manifest_json = json.dumps(manifest_data, sort_keys=True, default=str)
        manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()

        # Calculate bundle hash (hash of all component hashes)
        component_hashes = sorted([c.content_hash for c in self._manifest.components])
        bundle_content = "|".join(component_hashes) + "|" + manifest_hash
        bundle_hash = hashlib.sha256(bundle_content.encode()).hexdigest()

        # Update manifest
        now = datetime.now(timezone.utc)
        from datetime import timedelta
        retention_expires = now + timedelta(days=365 * self._manifest.retention_years)

        self._manifest.status = BundleStatus.SEALED
        self._manifest.sealed_at = now
        self._manifest.manifest_hash = manifest_hash
        self._manifest.bundle_hash = bundle_hash
        self._manifest.retention_expires = retention_expires

        self._sealed = True

        # Store if path configured
        if self._storage_path:
            self._persist_bundle()

        logger.info(
            f"Bundle {self._bundle_id} sealed: {len(self._manifest.components)} components, "
            f"hash={bundle_hash[:16]}..."
        )

        return self._manifest

    def get_manifest(self) -> BundleManifest:
        """Get current manifest."""
        return self._manifest.model_copy()

    def _add_component(
        self,
        component_type: ComponentType,
        name: str,
        data: Any,
        metadata: Dict[str, Any]
    ) -> str:
        """Add a component to the bundle."""
        if self._sealed:
            raise ValueError("Cannot add to sealed bundle")

        component_id = f"{component_type.value}_{name}_{uuid.uuid4().hex[:8]}"

        # Serialize data
        json_data = json.dumps(data, sort_keys=True, default=str)
        data_bytes = json_data.encode('utf-8')

        # Compress
        compressed_bytes = gzip.compress(data_bytes, compresslevel=6)

        # Calculate hash
        content_hash = hashlib.sha256(data_bytes).hexdigest()

        # Create component
        component = BundleComponent(
            component_id=component_id,
            component_type=component_type,
            filename=f"{component_id}.json.gz",
            content_hash=content_hash,
            size_bytes=len(compressed_bytes),
            compressed=True,
            metadata=metadata
        )

        self._manifest.components.append(component)
        self._components_data[component_id] = compressed_bytes

        logger.debug(f"Added component {component_id}: {len(compressed_bytes)} bytes, hash={content_hash[:16]}...")

        return component_id

    def _persist_bundle(self) -> None:
        """Persist bundle to storage."""
        if not self._storage_path:
            return

        bundle_path = self._storage_path / self._bundle_id
        bundle_path.mkdir(parents=True, exist_ok=True)

        # Write manifest
        manifest_path = bundle_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self._manifest.model_dump(), f, indent=2, default=str)

        # Write components
        for component in self._manifest.components:
            component_path = bundle_path / component.filename
            with open(component_path, 'wb') as f:
                f.write(self._components_data[component.component_id])

        logger.info(f"Bundle persisted to {bundle_path}")


class ImmutableStorage:
    """
    Content-addressed immutable storage for run bundles.

    Provides storage interface with SHA-256 content addressing
    and integrity verification.
    """

    def __init__(self, base_path: str):
        """Initialize storage."""
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ImmutableStorage initialized at {base_path}")

    def store_bundle(self, manifest: BundleManifest, components: Dict[str, bytes]) -> str:
        """
        Store a bundle in content-addressed storage.

        Args:
            manifest: Bundle manifest
            components: Component data keyed by component_id

        Returns:
            Storage path
        """
        if manifest.status != BundleStatus.SEALED:
            raise ValueError("Can only store sealed bundles")

        # Use bundle hash as directory name for content addressing
        bundle_dir = self._base_path / manifest.bundle_hash[:2] / manifest.bundle_hash
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Write manifest
        manifest_path = bundle_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest.model_dump(), f, indent=2, default=str)

        # Write components
        for component in manifest.components:
            if component.component_id in components:
                component_path = bundle_dir / component.filename
                with open(component_path, 'wb') as f:
                    f.write(components[component.component_id])

        logger.info(f"Bundle {manifest.bundle_id} stored at {bundle_dir}")
        return str(bundle_dir)

    def retrieve_bundle(self, bundle_hash: str) -> Optional[BundleManifest]:
        """
        Retrieve a bundle by its hash.

        Args:
            bundle_hash: SHA-256 hash of bundle

        Returns:
            Bundle manifest if found, None otherwise
        """
        bundle_dir = self._base_path / bundle_hash[:2] / bundle_hash
        manifest_path = bundle_dir / "manifest.json"

        if not manifest_path.exists():
            return None

        with open(manifest_path, 'r') as f:
            data = json.load(f)

        return BundleManifest(**data)

    def verify_integrity(self, bundle_hash: str) -> bool:
        """
        Verify integrity of a stored bundle.

        Args:
            bundle_hash: Bundle hash to verify

        Returns:
            True if bundle is intact, False otherwise
        """
        bundle_dir = self._base_path / bundle_hash[:2] / bundle_hash

        if not bundle_dir.exists():
            return False

        manifest = self.retrieve_bundle(bundle_hash)
        if manifest is None:
            return False

        # Verify each component
        for component in manifest.components:
            component_path = bundle_dir / component.filename

            if not component_path.exists():
                logger.error(f"Missing component: {component.filename}")
                return False

            # Decompress and verify hash
            with open(component_path, 'rb') as f:
                compressed = f.read()

            try:
                decompressed = gzip.decompress(compressed)
                actual_hash = hashlib.sha256(decompressed).hexdigest()

                if actual_hash != component.content_hash:
                    logger.error(f"Hash mismatch for {component.filename}")
                    return False

            except Exception as e:
                logger.error(f"Failed to verify {component.filename}: {e}")
                return False

        return True


class BundleReplayValidator:
    """Validates that bundle replays produce identical results."""

    def validate_replay(
        self,
        original_bundle: BundleManifest,
        replay_outputs: Dict[str, Any]
    ) -> ReplayValidationResult:
        """
        Validate that replay outputs match original bundle.

        Args:
            original_bundle: Original bundle manifest
            replay_outputs: Outputs from replay execution

        Returns:
            Validation result
        """
        validation_id = hashlib.sha256(
            f"replay|{original_bundle.bundle_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        mismatches = []
        components_validated = 0
        components_matched = 0

        # Find output components in original bundle
        output_components = [
            c for c in original_bundle.components
            if c.component_type == ComponentType.OUTPUT_DATA
        ]

        for component in output_components:
            components_validated += 1

            # Get corresponding replay output
            output_name = component.component_id.split('_')[2]  # Extract name from component_id

            if output_name not in replay_outputs:
                mismatches.append({
                    "component_id": component.component_id,
                    "error": "Missing in replay outputs",
                    "expected_hash": component.content_hash
                })
                continue

            # Hash replay output
            replay_data = json.dumps(replay_outputs[output_name], sort_keys=True, default=str)
            replay_hash = hashlib.sha256(replay_data.encode()).hexdigest()

            if replay_hash == component.content_hash:
                components_matched += 1
            else:
                mismatches.append({
                    "component_id": component.component_id,
                    "expected_hash": component.content_hash,
                    "actual_hash": replay_hash
                })

        is_identical = len(mismatches) == 0

        return ReplayValidationResult(
            bundle_id=original_bundle.bundle_id,
            validation_id=validation_id,
            is_identical=is_identical,
            components_validated=components_validated,
            components_matched=components_matched,
            mismatches=mismatches,
            original_hash=original_bundle.bundle_hash or "",
            replay_hash=hashlib.sha256(
                json.dumps(replay_outputs, sort_keys=True, default=str).encode()
            ).hexdigest()
        )
