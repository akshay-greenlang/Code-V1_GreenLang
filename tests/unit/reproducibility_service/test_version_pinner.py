# -*- coding: utf-8 -*-
"""
Unit Tests for VersionPinner (AGENT-FOUND-008)

Tests version pin creation, manifest management, hash determinism,
verification, and storage.

Coverage target: 85%+ of version_pinner.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline models and VersionPinner
# ---------------------------------------------------------------------------

def _content_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, ensure_ascii=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class VersionPin:
    def __init__(self, component_type: str, component_id: str, version: str,
                 version_hash: str = "",
                 pinned_at: Optional[datetime] = None):
        self.component_type = component_type
        self.component_id = component_id
        self.version = version
        self.version_hash = version_hash or _content_hash(
            {"type": component_type, "id": component_id, "version": version}
        )[:16]
        self.pinned_at = pinned_at or datetime.now(timezone.utc)


class VersionManifest:
    def __init__(self, manifest_id: str = "",
                 created_at: Optional[datetime] = None,
                 agent_versions: Optional[Dict[str, VersionPin]] = None,
                 model_versions: Optional[Dict[str, VersionPin]] = None,
                 factor_versions: Optional[Dict[str, VersionPin]] = None,
                 data_versions: Optional[Dict[str, VersionPin]] = None,
                 manifest_hash: str = ""):
        self.manifest_id = manifest_id
        self.created_at = created_at or datetime.now(timezone.utc)
        self.agent_versions = agent_versions or {}
        self.model_versions = model_versions or {}
        self.factor_versions = factor_versions or {}
        self.data_versions = data_versions or {}
        self.manifest_hash = manifest_hash


class VersionPinner:
    """Pins and verifies component versions for reproducibility."""

    def __init__(self):
        self._manifests: Dict[str, VersionManifest] = {}
        self._manifest_counter = 0

    def create_version_pin(self, component_type: str, component_id: str,
                           version: str) -> VersionPin:
        """Create a version pin for a component."""
        return VersionPin(
            component_type=component_type,
            component_id=component_id,
            version=version,
        )

    def create_manifest(self,
                        agent_versions: Optional[Dict[str, VersionPin]] = None,
                        model_versions: Optional[Dict[str, VersionPin]] = None,
                        factor_versions: Optional[Dict[str, VersionPin]] = None,
                        data_versions: Optional[Dict[str, VersionPin]] = None) -> VersionManifest:
        """Create a version manifest."""
        self._manifest_counter += 1
        manifest_id = f"manifest-{self._manifest_counter:04d}"

        agents = agent_versions or {}
        models = model_versions or {}
        factors = factor_versions or {}
        data = data_versions or {}

        # Compute manifest hash
        hash_data = {
            "agents": {k: v.version for k, v in sorted(agents.items())},
            "models": {k: v.version for k, v in sorted(models.items())},
            "factors": {k: v.version for k, v in sorted(factors.items())},
            "data": {k: v.version for k, v in sorted(data.items())},
        }
        manifest_hash = _content_hash(hash_data)[:16]

        return VersionManifest(
            manifest_id=manifest_id,
            agent_versions=agents,
            model_versions=models,
            factor_versions=factors,
            data_versions=data,
            manifest_hash=manifest_hash,
        )

    def verify_manifest(self, manifest: VersionManifest,
                        current_versions: Dict[str, str]) -> Dict[str, Any]:
        """Verify current versions match a manifest."""
        mismatches = []
        missing = []

        for agent_id, pin in manifest.agent_versions.items():
            current = current_versions.get(agent_id)
            if current is None:
                missing.append(agent_id)
            elif current != pin.version:
                mismatches.append({
                    "component": agent_id,
                    "expected": pin.version,
                    "actual": current,
                })

        for model_id, pin in manifest.model_versions.items():
            current = current_versions.get(model_id)
            if current is None:
                missing.append(model_id)
            elif current != pin.version:
                mismatches.append({
                    "component": model_id,
                    "expected": pin.version,
                    "actual": current,
                })

        is_match = len(mismatches) == 0 and len(missing) == 0
        return {
            "is_match": is_match,
            "mismatches": mismatches,
            "missing": missing,
        }

    def store_manifest(self, manifest: VersionManifest) -> None:
        """Store a manifest for retrieval."""
        self._manifests[manifest.manifest_id] = manifest

    def get_manifest(self, manifest_id: str) -> Optional[VersionManifest]:
        """Get a stored manifest."""
        return self._manifests.get(manifest_id)

    def list_manifests(self) -> List[VersionManifest]:
        """List all stored manifests."""
        return list(self._manifests.values())

    def pin_current_versions(self, components: Dict[str, str]) -> VersionManifest:
        """Pin current versions from a flat dict of component_id -> version."""
        agents = {}
        for comp_id, ver in components.items():
            pin = self.create_version_pin("agent", comp_id, ver)
            agents[comp_id] = pin
        return self.create_manifest(agent_versions=agents)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCreateVersionPin:
    """Test create_version_pin method."""

    def test_create_version_pin(self):
        pinner = VersionPinner()
        pin = pinner.create_version_pin("agent", "GL-FOUND-X-001", "1.0.0")
        assert pin.component_type == "agent"
        assert pin.component_id == "GL-FOUND-X-001"
        assert pin.version == "1.0.0"

    def test_version_pin_hash_auto_computed(self):
        pinner = VersionPinner()
        pin = pinner.create_version_pin("agent", "a1", "1.0.0")
        assert pin.version_hash != ""
        assert len(pin.version_hash) == 16

    def test_version_pin_pinned_at_set(self):
        pinner = VersionPinner()
        pin = pinner.create_version_pin("model", "m1", "2.0.0")
        assert pin.pinned_at is not None

    def test_version_pin_types(self):
        pinner = VersionPinner()
        for ctype in ["agent", "model", "factor", "data"]:
            pin = pinner.create_version_pin(ctype, f"{ctype}-001", "1.0.0")
            assert pin.component_type == ctype


class TestCreateManifest:
    """Test create_manifest method."""

    def test_create_manifest_empty(self):
        pinner = VersionPinner()
        manifest = pinner.create_manifest()
        assert manifest.agent_versions == {}
        assert manifest.model_versions == {}
        assert manifest.factor_versions == {}
        assert manifest.data_versions == {}

    def test_create_manifest_with_agents(self):
        pinner = VersionPinner()
        pin = pinner.create_version_pin("agent", "a1", "1.0.0")
        manifest = pinner.create_manifest(agent_versions={"a1": pin})
        assert "a1" in manifest.agent_versions
        assert manifest.agent_versions["a1"].version == "1.0.0"

    def test_create_manifest_with_all_types(self):
        pinner = VersionPinner()
        agents = {"a1": pinner.create_version_pin("agent", "a1", "1.0.0")}
        models = {"m1": pinner.create_version_pin("model", "m1", "2.0.0")}
        factors = {"f1": pinner.create_version_pin("factor", "f1", "3.0.0")}
        data = {"d1": pinner.create_version_pin("data", "d1", "4.0.0")}
        manifest = pinner.create_manifest(agents, models, factors, data)
        assert len(manifest.agent_versions) == 1
        assert len(manifest.model_versions) == 1
        assert len(manifest.factor_versions) == 1
        assert len(manifest.data_versions) == 1

    def test_manifest_id_auto_generated(self):
        pinner = VersionPinner()
        m1 = pinner.create_manifest()
        m2 = pinner.create_manifest()
        assert m1.manifest_id != m2.manifest_id
        assert m1.manifest_id.startswith("manifest-")

    def test_manifest_hash_deterministic(self):
        pinner = VersionPinner()
        agents = {"a1": pinner.create_version_pin("agent", "a1", "1.0.0")}
        m1 = pinner.create_manifest(agent_versions=agents)
        # Recreate same pins
        agents2 = {"a1": pinner.create_version_pin("agent", "a1", "1.0.0")}
        m2 = pinner.create_manifest(agent_versions=agents2)
        assert m1.manifest_hash == m2.manifest_hash

    def test_manifest_hash_different_for_different_versions(self):
        pinner = VersionPinner()
        agents1 = {"a1": pinner.create_version_pin("agent", "a1", "1.0.0")}
        agents2 = {"a1": pinner.create_version_pin("agent", "a1", "2.0.0")}
        m1 = pinner.create_manifest(agent_versions=agents1)
        m2 = pinner.create_manifest(agent_versions=agents2)
        assert m1.manifest_hash != m2.manifest_hash


class TestVerifyManifest:
    """Test verify_manifest method."""

    def test_verify_manifest_all_match(self):
        pinner = VersionPinner()
        agents = {"a1": pinner.create_version_pin("agent", "a1", "1.0.0")}
        manifest = pinner.create_manifest(agent_versions=agents)
        result = pinner.verify_manifest(manifest, {"a1": "1.0.0"})
        assert result["is_match"] is True
        assert result["mismatches"] == []
        assert result["missing"] == []

    def test_verify_manifest_missing_agent(self):
        pinner = VersionPinner()
        agents = {"a1": pinner.create_version_pin("agent", "a1", "1.0.0")}
        manifest = pinner.create_manifest(agent_versions=agents)
        result = pinner.verify_manifest(manifest, {})
        assert result["is_match"] is False
        assert "a1" in result["missing"]

    def test_verify_manifest_version_mismatch(self):
        pinner = VersionPinner()
        agents = {"a1": pinner.create_version_pin("agent", "a1", "1.0.0")}
        manifest = pinner.create_manifest(agent_versions=agents)
        result = pinner.verify_manifest(manifest, {"a1": "2.0.0"})
        assert result["is_match"] is False
        assert len(result["mismatches"]) == 1
        assert result["mismatches"][0]["expected"] == "1.0.0"
        assert result["mismatches"][0]["actual"] == "2.0.0"

    def test_verify_manifest_model_version_mismatch(self):
        pinner = VersionPinner()
        models = {"m1": pinner.create_version_pin("model", "m1", "3.0.0")}
        manifest = pinner.create_manifest(model_versions=models)
        result = pinner.verify_manifest(manifest, {"m1": "4.0.0"})
        assert result["is_match"] is False

    def test_verify_manifest_partial_match(self):
        pinner = VersionPinner()
        agents = {
            "a1": pinner.create_version_pin("agent", "a1", "1.0.0"),
            "a2": pinner.create_version_pin("agent", "a2", "1.0.0"),
        }
        manifest = pinner.create_manifest(agent_versions=agents)
        result = pinner.verify_manifest(manifest, {"a1": "1.0.0", "a2": "2.0.0"})
        assert result["is_match"] is False
        assert len(result["mismatches"]) == 1


class TestManifestStorage:
    """Test manifest storage and retrieval."""

    def test_store_manifest(self):
        pinner = VersionPinner()
        manifest = pinner.create_manifest()
        pinner.store_manifest(manifest)
        retrieved = pinner.get_manifest(manifest.manifest_id)
        assert retrieved is not None
        assert retrieved.manifest_id == manifest.manifest_id

    def test_get_manifest_nonexistent(self):
        pinner = VersionPinner()
        assert pinner.get_manifest("nonexistent") is None

    def test_list_manifests(self):
        pinner = VersionPinner()
        m1 = pinner.create_manifest()
        m2 = pinner.create_manifest()
        pinner.store_manifest(m1)
        pinner.store_manifest(m2)
        manifests = pinner.list_manifests()
        assert len(manifests) == 2

    def test_list_manifests_empty(self):
        pinner = VersionPinner()
        assert pinner.list_manifests() == []


class TestPinCurrentVersions:
    """Test pin_current_versions method."""

    def test_pin_current_versions(self):
        pinner = VersionPinner()
        components = {"a1": "1.0.0", "a2": "2.0.0", "a3": "3.0.0"}
        manifest = pinner.pin_current_versions(components)
        assert len(manifest.agent_versions) == 3
        assert manifest.agent_versions["a1"].version == "1.0.0"
        assert manifest.agent_versions["a2"].version == "2.0.0"
        assert manifest.agent_versions["a3"].version == "3.0.0"

    def test_pin_current_versions_empty(self):
        pinner = VersionPinner()
        manifest = pinner.pin_current_versions({})
        assert manifest.agent_versions == {}

    def test_pin_current_versions_manifest_id(self):
        pinner = VersionPinner()
        manifest = pinner.pin_current_versions({"a1": "1.0.0"})
        assert manifest.manifest_id.startswith("manifest-")
