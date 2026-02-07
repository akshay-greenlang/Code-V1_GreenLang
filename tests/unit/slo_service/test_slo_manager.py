# -*- coding: utf-8 -*-
"""
Unit tests for SLO Manager (OBS-005)

Tests CRUD operations, YAML import/export, version history, concurrent
access safety, and name uniqueness enforcement.

Coverage target: 85%+ of slo_manager.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from greenlang.infrastructure.slo_service.models import SLI, SLO, SLIType
from greenlang.infrastructure.slo_service.slo_manager import SLOManager


class TestSLOManagerCRUD:
    """Tests for SLO CRUD operations."""

    def test_create_slo_in_registry(self, sample_slo):
        """SLO is stored in the registry after creation."""
        mgr = SLOManager()
        created = mgr.create(sample_slo)
        assert created.slo_id == sample_slo.slo_id
        assert mgr.get(sample_slo.slo_id) is not None

    def test_create_duplicate_id_raises(self, sample_slo):
        """Creating two SLOs with same ID raises ValueError."""
        mgr = SLOManager()
        mgr.create(sample_slo)

        with pytest.raises(ValueError, match="already exists"):
            mgr.create(sample_slo)

    def test_slo_name_uniqueness(self, slo_factory):
        """Creating two SLOs with the same name raises ValueError."""
        mgr = SLOManager()
        mgr.create(slo_factory(slo_id="slo-1", name="Same Name"))

        with pytest.raises(ValueError, match="already exists"):
            mgr.create(slo_factory(slo_id="slo-2", name="Same Name"))

    def test_get_slo_by_id(self, sample_slo):
        """SLO can be retrieved by ID."""
        mgr = SLOManager()
        mgr.create(sample_slo)
        result = mgr.get(sample_slo.slo_id)
        assert result is not None
        assert result.slo_id == sample_slo.slo_id

    def test_get_slo_not_found(self):
        """Returns None for non-existent SLO ID."""
        mgr = SLOManager()
        assert mgr.get("nonexistent") is None

    def test_update_slo_version_increment(self, sample_slo):
        """Updating an SLO increments its version."""
        mgr = SLOManager()
        mgr.create(sample_slo)
        assert sample_slo.version == 1

        updated = mgr.update(sample_slo.slo_id, {"description": "Updated"})
        assert updated.version == 2
        assert updated.description == "Updated"

    def test_update_preserves_slo_id(self, sample_slo):
        """slo_id cannot be changed via update."""
        mgr = SLOManager()
        mgr.create(sample_slo)
        mgr.update(sample_slo.slo_id, {"slo_id": "new-id"})
        assert mgr.get(sample_slo.slo_id) is not None

    def test_update_nonexistent_raises(self):
        """Updating a non-existent SLO raises KeyError."""
        mgr = SLOManager()
        with pytest.raises(KeyError):
            mgr.update("nonexistent", {"name": "New"})

    def test_delete_slo_soft_delete(self, sample_slo):
        """Deleting an SLO sets deleted=True but keeps it in registry."""
        mgr = SLOManager()
        mgr.create(sample_slo)
        result = mgr.delete(sample_slo.slo_id)
        assert result is True
        assert mgr.get(sample_slo.slo_id) is None  # hidden from get

    def test_delete_nonexistent_returns_false(self):
        """Deleting a non-existent SLO returns False."""
        mgr = SLOManager()
        assert mgr.delete("nonexistent") is False

    def test_list_slos_all(self, sample_slo_list):
        """List all SLOs returns all non-deleted."""
        mgr = SLOManager()
        for slo in sample_slo_list:
            mgr.create(slo)
        result = mgr.list_all()
        assert len(result) == len(sample_slo_list)

    def test_list_slos_by_service(self, sample_slo_list):
        """Filter SLOs by service name."""
        mgr = SLOManager()
        for slo in sample_slo_list:
            mgr.create(slo)
        result = mgr.list_all(service="api-gateway")
        assert all(s.service == "api-gateway" for s in result)

    def test_list_slos_by_team(self, sample_slo_list):
        """Filter SLOs by team name."""
        mgr = SLOManager()
        for slo in sample_slo_list:
            mgr.create(slo)
        result = mgr.list_all(team="data-platform")
        assert len(result) == 1
        assert result[0].team == "data-platform"

    def test_get_slo_history(self, sample_slo):
        """Version history is preserved after updates."""
        mgr = SLOManager()
        mgr.create(sample_slo)
        mgr.update(sample_slo.slo_id, {"description": "v2"})
        mgr.update(sample_slo.slo_id, {"description": "v3"})

        history = mgr.get_history(sample_slo.slo_id)
        assert len(history) == 2
        assert history[0]["description"] != history[1]["description"]


class TestSLOManagerYAML:
    """Tests for YAML import/export."""

    def test_load_from_yaml_valid(self, tmp_path, sample_slo_yaml_data):
        """Load SLOs from a valid YAML file."""
        yaml_file = tmp_path / "slos.yaml"
        import yaml
        yaml_file.write_text(yaml.dump(sample_slo_yaml_data))

        mgr = SLOManager()
        loaded = mgr.load_from_yaml(str(yaml_file))
        assert len(loaded) == 2

    def test_load_from_yaml_invalid_path(self):
        """Loading from non-existent path raises FileNotFoundError."""
        mgr = SLOManager()
        with pytest.raises(FileNotFoundError):
            mgr.load_from_yaml("/nonexistent/path.yaml")

    def test_load_from_yaml_invalid_format(self, tmp_path):
        """Loading invalid YAML format raises ValueError."""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("just_a_string: true")

        mgr = SLOManager()
        with pytest.raises(ValueError, match="expected 'slos' key"):
            mgr.load_from_yaml(str(yaml_file))

    def test_save_to_yaml(self, tmp_path, sample_slo):
        """Export SLOs to YAML file."""
        mgr = SLOManager()
        mgr.create(sample_slo)

        output_path = str(tmp_path / "export.yaml")
        mgr.save_to_yaml(output_path)

        assert Path(output_path).exists()
        import yaml
        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert "slos" in data
        assert len(data["slos"]) == 1

    def test_import_export_roundtrip(self, tmp_path, sample_slo_yaml_data):
        """SLOs survive YAML import -> export round-trip."""
        import yaml

        input_file = tmp_path / "input.yaml"
        input_file.write_text(yaml.dump(sample_slo_yaml_data))

        mgr = SLOManager()
        mgr.load_from_yaml(str(input_file))

        output_file = str(tmp_path / "output.yaml")
        mgr.save_to_yaml(output_file)

        with open(output_file) as f:
            exported = yaml.safe_load(f)

        assert len(exported["slos"]) == len(sample_slo_yaml_data["slos"])

    def test_slo_registry_reload(self, tmp_path, sample_slo_yaml_data):
        """Reload clears registry and re-imports from YAML."""
        import yaml

        yaml_file = tmp_path / "slos.yaml"
        yaml_file.write_text(yaml.dump(sample_slo_yaml_data))

        mgr = SLOManager()
        mgr.load_from_yaml(str(yaml_file))
        assert len(mgr.list_all()) == 2

        reloaded = mgr.reload(str(yaml_file))
        assert len(reloaded) == 2


class TestSLOManagerConcurrency:
    """Tests for concurrent access safety."""

    def test_concurrent_access_safety(self, slo_factory):
        """Multiple threads can safely create SLOs concurrently."""
        mgr = SLOManager()
        errors = []

        def create_slo(idx):
            try:
                slo = slo_factory(
                    slo_id=f"concurrent-{idx}",
                    name=f"Concurrent SLO {idx}",
                )
                mgr.create(slo)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=create_slo, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(mgr.list_all()) == 20

    def test_slo_validation_on_create(self, slo_factory):
        """SLO create validates uniqueness atomically."""
        mgr = SLOManager()
        mgr.create(slo_factory(slo_id="unique-1", name="Unique SLO"))

        with pytest.raises(ValueError):
            mgr.create(slo_factory(slo_id="unique-1", name="Duplicate ID"))
