"""
Comprehensive tests for SDK Dataset abstraction.

Tests cover:
- Dataset initialization
- Load and save operations
- Dataset description
- Schema and statistics
- Dataset card
- Metadata handling
"""

import pytest
import json
from pathlib import Path
from typing import Any, Dict, Optional
from greenlang.sdk.base import Dataset, Metadata


class MockDataset(Dataset):
    """Mock dataset for testing."""

    def __init__(self, metadata: Optional[Metadata] = None):
        """Initialize mock dataset."""
        super().__init__(metadata)
        self.data: Optional[Any] = None
        self.load_count = 0
        self.save_count = 0

    def load(self) -> Any:
        """Load and return mock data."""
        self.load_count += 1
        self.data = {"records": [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]}
        return self.data

    def save(self, data: Any) -> bool:
        """Save data."""
        self.save_count += 1
        self.data = data
        return True

    def describe(self) -> Dict[str, Any]:
        """Describe the dataset."""
        return {
            "name": self.metadata.name,
            "records": len(self.data.get("records", [])) if self.data else 0,
            "schema": self.get_schema(),
        }


class FileDataset(Dataset):
    """Dataset that works with files."""

    def __init__(self, path: Path, metadata: Optional[Metadata] = None):
        """Initialize file dataset."""
        super().__init__(metadata)
        self.path = path
        self.data = None

    def load(self) -> Any:
        """Load data from file."""
        if not self.path.exists():
            return None

        with open(self.path) as f:
            self.data = json.load(f)
        return self.data

    def save(self, data: Any) -> bool:
        """Save data to file."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(data, f, indent=2)
            self.data = data
            return True
        except Exception as e:
            self.logger.error(f"Failed to save: {e}")
            return False

    def describe(self) -> Dict[str, Any]:
        """Describe the dataset."""
        return {
            "name": self.metadata.name,
            "path": str(self.path),
            "exists": self.path.exists(),
            "size": self.path.stat().st_size if self.path.exists() else 0,
        }

    def get_schema(self) -> Dict[str, Any]:
        """Get data schema."""
        return {"type": "object", "format": "json"}

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.data:
            return {}
        return {"records": len(self.data) if isinstance(self.data, list) else 1}


@pytest.mark.unit
class TestDatasetInitialization:
    """Test Dataset initialization."""

    def test_dataset_default_init(self):
        """Test creating dataset with defaults."""
        dataset = MockDataset()

        assert dataset.metadata is not None
        assert dataset.metadata.id == "mockdataset"
        assert dataset.metadata.name == "MockDataset"
        assert dataset.logger is not None

    def test_dataset_with_custom_metadata(self):
        """Test creating dataset with custom metadata."""
        metadata = Metadata(
            id="custom-dataset",
            name="Custom Dataset",
            version="2.0.0",
            description="Test dataset",
        )
        dataset = MockDataset(metadata=metadata)

        assert dataset.metadata.id == "custom-dataset"
        assert dataset.metadata.name == "Custom Dataset"
        assert dataset.metadata.version == "2.0.0"


@pytest.mark.unit
class TestDatasetLoad:
    """Test dataset load operations."""

    def test_load_data(self):
        """Test loading data from dataset."""
        dataset = MockDataset()
        data = dataset.load()

        assert data is not None
        assert "records" in data
        assert len(data["records"]) == 2
        assert dataset.load_count == 1

    def test_load_multiple_times(self):
        """Test loading data multiple times."""
        dataset = MockDataset()

        data1 = dataset.load()
        data2 = dataset.load()

        assert dataset.load_count == 2
        assert data1 is not None
        assert data2 is not None

    def test_load_from_file(self, tmp_path):
        """Test loading data from file."""
        data_file = tmp_path / "data.json"
        test_data = {"records": [{"id": 1}, {"id": 2}]}
        data_file.write_text(json.dumps(test_data))

        dataset = FileDataset(data_file)
        loaded_data = dataset.load()

        assert loaded_data == test_data

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file."""
        data_file = tmp_path / "missing.json"
        dataset = FileDataset(data_file)

        loaded_data = dataset.load()

        assert loaded_data is None


@pytest.mark.unit
class TestDatasetSave:
    """Test dataset save operations."""

    def test_save_data(self):
        """Test saving data to dataset."""
        dataset = MockDataset()
        data = {"test": "data"}

        result = dataset.save(data)

        assert result is True
        assert dataset.data == data
        assert dataset.save_count == 1

    def test_save_multiple_times(self):
        """Test saving data multiple times."""
        dataset = MockDataset()

        dataset.save({"first": "save"})
        dataset.save({"second": "save"})

        assert dataset.save_count == 2
        assert dataset.data == {"second": "save"}

    def test_save_to_file(self, tmp_path):
        """Test saving data to file."""
        data_file = tmp_path / "output.json"
        dataset = FileDataset(data_file)
        test_data = {"records": [1, 2, 3]}

        result = dataset.save(test_data)

        assert result is True
        assert data_file.exists()

        # Verify content
        with open(data_file) as f:
            saved_data = json.load(f)
        assert saved_data == test_data

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates directory if needed."""
        data_file = tmp_path / "nested" / "dir" / "data.json"
        dataset = FileDataset(data_file)

        result = dataset.save({"test": "data"})

        assert result is True
        assert data_file.parent.exists()
        assert data_file.exists()


@pytest.mark.unit
class TestDatasetDescribe:
    """Test dataset description."""

    def test_describe_basic(self):
        """Test basic dataset description."""
        dataset = MockDataset()
        dataset.load()

        description = dataset.describe()

        assert isinstance(description, dict)
        assert "name" in description
        assert "records" in description
        assert description["records"] == 2

    def test_describe_file_dataset(self, tmp_path):
        """Test describing file dataset."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps({"test": "data"}))

        dataset = FileDataset(data_file)
        description = dataset.describe()

        assert "name" in description
        assert "path" in description
        assert description["exists"] is True
        assert description["size"] > 0

    def test_describe_empty_dataset(self):
        """Test describing empty dataset."""
        dataset = MockDataset()

        description = dataset.describe()

        assert description["records"] == 0


@pytest.mark.unit
class TestDatasetSchema:
    """Test dataset schema methods."""

    def test_get_schema_default(self):
        """Test getting default schema."""
        dataset = MockDataset()
        schema = dataset.get_schema()

        assert isinstance(schema, dict)
        assert schema["type"] == "object"

    def test_get_schema_custom(self, tmp_path):
        """Test getting custom schema."""
        data_file = tmp_path / "data.json"
        dataset = FileDataset(data_file)

        schema = dataset.get_schema()

        assert schema["type"] == "object"
        assert schema["format"] == "json"


@pytest.mark.unit
class TestDatasetStats:
    """Test dataset statistics."""

    def test_get_stats_default(self):
        """Test getting default stats."""
        dataset = MockDataset()
        stats = dataset.get_stats()

        assert isinstance(stats, dict)

    def test_get_stats_with_data(self, tmp_path):
        """Test getting stats with data."""
        data_file = tmp_path / "data.json"
        dataset = FileDataset(data_file)
        dataset.save([1, 2, 3, 4, 5])

        stats = dataset.get_stats()

        assert "records" in stats
        assert stats["records"] == 5

    def test_get_stats_no_data(self, tmp_path):
        """Test getting stats when no data loaded."""
        data_file = tmp_path / "data.json"
        dataset = FileDataset(data_file)

        stats = dataset.get_stats()

        assert stats == {}


@pytest.mark.unit
class TestDatasetCard:
    """Test dataset card functionality."""

    def test_get_card_default(self):
        """Test getting default card."""
        dataset = MockDataset()
        card = dataset.get_card()

        assert card is None

    def test_get_card_custom(self):
        """Test custom dataset card implementation."""

        class DocumentedDataset(MockDataset):
            def get_card(self) -> str:
                return "# Dataset Card\n\nThis is a test dataset."

        dataset = DocumentedDataset()
        card = dataset.get_card()

        assert card is not None
        assert "Dataset Card" in card


@pytest.mark.unit
class TestDatasetLoadAndSave:
    """Test combined load and save operations."""

    def test_load_modify_save(self, tmp_path):
        """Test loading, modifying, and saving data."""
        data_file = tmp_path / "data.json"
        initial_data = {"value": 1}
        data_file.write_text(json.dumps(initial_data))

        dataset = FileDataset(data_file)

        # Load
        loaded = dataset.load()
        assert loaded["value"] == 1

        # Modify
        loaded["value"] = 2

        # Save
        dataset.save(loaded)

        # Load again to verify
        dataset2 = FileDataset(data_file)
        final_data = dataset2.load()
        assert final_data["value"] == 2

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test roundtrip: save then load."""
        data_file = tmp_path / "roundtrip.json"
        dataset = FileDataset(data_file)

        original_data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        # Save
        dataset.save(original_data)

        # Load
        loaded_data = dataset.load()

        assert loaded_data == original_data


@pytest.mark.unit
class TestDatasetMetadata:
    """Test dataset metadata handling."""

    def test_metadata_in_description(self):
        """Test that metadata appears in description."""
        metadata = Metadata(
            id="test-dataset",
            name="Test Dataset",
            version="1.0.0",
            tags=["test", "mock"],
        )
        dataset = MockDataset(metadata=metadata)

        description = dataset.describe()

        assert description["name"] == "Test Dataset"

    def test_metadata_timestamps(self):
        """Test metadata timestamps."""
        dataset = MockDataset()

        assert dataset.metadata.created_at is not None
        assert dataset.metadata.updated_at is not None
