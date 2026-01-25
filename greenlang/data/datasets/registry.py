"""
Dataset Registry
================

Registry for managing and accessing datasets.

Author: Data Team
Created: 2025-11-21
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for a registered dataset."""
    name: str
    version: str
    description: str
    created_at: datetime
    updated_at: datetime
    source: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    checksum: Optional[str] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    size_bytes: Optional[int] = None


@dataclass
class Dataset:
    """A registered dataset with data and metadata."""
    data: Any
    metadata: DatasetMetadata
    transformations: List[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def access(self) -> Any:
        """Access the dataset and update counters."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        return self.data


class DatasetRegistry:
    """
    Central registry for datasets used in GreenLang applications.

    Manages dataset registration, versioning, and access control.
    """

    def __init__(self):
        """Initialize dataset registry."""
        self.datasets: Dict[str, Dataset] = {}
        self.versions: Dict[str, List[str]] = {}  # name -> list of versions
        self._locked_datasets: set = set()

    def register(
        self,
        name: str,
        data: Any,
        version: str = "1.0.0",
        description: str = "",
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None,
        calculate_checksum: bool = True
    ) -> Dataset:
        """
        Register a dataset in the registry.

        Args:
            name: Dataset name
            data: Dataset data
            version: Dataset version
            description: Dataset description
            source: Data source
            tags: Dataset tags
            schema: Data schema
            calculate_checksum: Whether to calculate data checksum

        Returns:
            Registered Dataset object
        """
        # Create unique key
        key = self._make_key(name, version)

        # Check if already registered
        if key in self.datasets:
            raise ValueError(f"Dataset {name} v{version} already registered")

        # Calculate metadata
        metadata = DatasetMetadata(
            name=name,
            version=version,
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source=source,
            schema=schema,
            tags=tags or []
        )

        # Calculate checksum if requested
        if calculate_checksum:
            metadata.checksum = self._calculate_checksum(data)

        # Extract data statistics
        metadata = self._extract_statistics(data, metadata)

        # Create dataset
        dataset = Dataset(data=data, metadata=metadata)

        # Register
        self.datasets[key] = dataset

        # Track versions
        if name not in self.versions:
            self.versions[name] = []
        self.versions[name].append(version)

        logger.info(f"Registered dataset: {name} v{version}")
        return dataset

    def get(self, name: str, version: Optional[str] = None) -> Optional[Dataset]:
        """
        Get a dataset from the registry.

        Args:
            name: Dataset name
            version: Dataset version (latest if None)

        Returns:
            Dataset or None if not found
        """
        # Get latest version if not specified
        if version is None:
            version = self.get_latest_version(name)
            if version is None:
                return None

        key = self._make_key(name, version)
        dataset = self.datasets.get(key)

        if dataset:
            dataset.access()
            logger.debug(f"Accessed dataset: {name} v{version}")

        return dataset

    def get_data(self, name: str, version: Optional[str] = None) -> Any:
        """
        Get dataset data directly.

        Args:
            name: Dataset name
            version: Dataset version

        Returns:
            Dataset data or None if not found
        """
        dataset = self.get(name, version)
        return dataset.data if dataset else None

    def list_datasets(self) -> List[str]:
        """List all registered dataset names."""
        return list(self.versions.keys())

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a dataset."""
        return self.versions.get(name, [])

    def get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version of a dataset."""
        versions = self.list_versions(name)
        if not versions:
            return None

        # Sort versions (assuming semantic versioning)
        try:
            sorted_versions = sorted(
                versions,
                key=lambda v: tuple(map(int, v.split('.')))
            )
            return sorted_versions[-1]
        except:
            # Fallback to string sorting
            return sorted(versions)[-1]

    def update(
        self,
        name: str,
        data: Any,
        version: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dataset:
        """
        Update an existing dataset (creates new version).

        Args:
            name: Dataset name
            data: New data
            version: New version (auto-incremented if None)
            description: Update description

        Returns:
            Updated Dataset object
        """
        # Get current version
        current_version = self.get_latest_version(name)
        if current_version is None:
            raise ValueError(f"Dataset {name} not found")

        # Auto-increment version if not provided
        if version is None:
            parts = current_version.split('.')
            parts[-1] = str(int(parts[-1]) + 1)
            version = '.'.join(parts)

        # Get current metadata
        current_dataset = self.get(name, current_version)
        if not current_dataset:
            raise ValueError(f"Dataset {name} v{current_version} not found")

        # Register new version
        return self.register(
            name=name,
            data=data,
            version=version,
            description=description or f"Updated from v{current_version}",
            source=current_dataset.metadata.source,
            tags=current_dataset.metadata.tags,
            schema=current_dataset.metadata.schema
        )

    def delete(self, name: str, version: Optional[str] = None) -> bool:
        """
        Delete a dataset from the registry.

        Args:
            name: Dataset name
            version: Dataset version (all versions if None)

        Returns:
            True if deleted, False if not found
        """
        if name in self._locked_datasets:
            raise ValueError(f"Dataset {name} is locked and cannot be deleted")

        if version is None:
            # Delete all versions
            versions = self.list_versions(name)
            for v in versions:
                key = self._make_key(name, v)
                if key in self.datasets:
                    del self.datasets[key]
            if name in self.versions:
                del self.versions[name]
            logger.info(f"Deleted all versions of dataset: {name}")
            return True
        else:
            # Delete specific version
            key = self._make_key(name, version)
            if key in self.datasets:
                del self.datasets[key]
                if name in self.versions and version in self.versions[name]:
                    self.versions[name].remove(version)
                logger.info(f"Deleted dataset: {name} v{version}")
                return True

        return False

    def lock(self, name: str) -> None:
        """Lock a dataset to prevent deletion."""
        self._locked_datasets.add(name)
        logger.info(f"Locked dataset: {name}")

    def unlock(self, name: str) -> None:
        """Unlock a dataset."""
        self._locked_datasets.discard(name)
        logger.info(f"Unlocked dataset: {name}")

    def get_metadata(self, name: str, version: Optional[str] = None) -> Optional[DatasetMetadata]:
        """Get metadata for a dataset."""
        dataset = self.get(name, version)
        return dataset.metadata if dataset else None

    def search(self, tags: Optional[List[str]] = None, source: Optional[str] = None) -> List[Dataset]:
        """
        Search for datasets by tags or source.

        Args:
            tags: Tags to search for
            source: Source to filter by

        Returns:
            List of matching datasets
        """
        results = []

        for dataset in self.datasets.values():
            if tags and not any(tag in dataset.metadata.tags for tag in tags):
                continue
            if source and dataset.metadata.source != source:
                continue
            results.append(dataset)

        return results

    def _make_key(self, name: str, version: str) -> str:
        """Create unique key for dataset."""
        return f"{name}:{version}"

    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data."""
        if hasattr(data, 'to_json'):
            # Pandas DataFrame
            data_str = data.to_json(orient='records')
        elif isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def _extract_statistics(self, data: Any, metadata: DatasetMetadata) -> DatasetMetadata:
        """Extract statistics from data."""
        try:
            if hasattr(data, 'shape'):
                # Pandas DataFrame
                metadata.row_count = data.shape[0]
                metadata.column_count = data.shape[1]
            elif isinstance(data, list):
                metadata.row_count = len(data)
            elif isinstance(data, dict):
                metadata.row_count = len(data)
        except:
            pass

        return metadata