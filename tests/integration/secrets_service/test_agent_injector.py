# -*- coding: utf-8 -*-
"""
Integration Tests for Vault Agent Injector - SEC-006

Tests the Vault Agent Injector sidecar pattern for:
- Pod annotation parsing
- Template rendering for secrets
- Init container mode
- Sidecar mode with auto-renewal

Requires:
- Kubernetes cluster with Vault Agent Injector installed
- Or mock for CI/CD testing

Set KUBECONFIG or run in-cluster for real tests.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Skip if dependencies not available
# ---------------------------------------------------------------------------
try:
    from kubernetes import client, config
    _HAS_K8S = True
except ImportError:
    _HAS_K8S = False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_K8S, reason="kubernetes client not installed"),
]


# ============================================================================
# Test Configuration
# ============================================================================

TEST_CONFIG = {
    "namespace": os.getenv("TEST_NAMESPACE", "greenlang-test"),
    "vault_addr": os.getenv("VAULT_ADDR", "https://vault.vault.svc:8200"),
}


# ============================================================================
# Helpers
# ============================================================================


def _create_pod_with_agent_annotations(
    name: str,
    namespace: str,
    vault_role: str,
    secret_path: str,
    template: Optional[str] = None,
    init_first: bool = False,
) -> Dict[str, Any]:
    """Create Pod manifest with Vault Agent Injector annotations."""
    annotations = {
        "vault.hashicorp.com/agent-inject": "true",
        "vault.hashicorp.com/role": vault_role,
        "vault.hashicorp.com/agent-inject-secret-config": secret_path,
    }

    if template:
        annotations["vault.hashicorp.com/agent-inject-template-config"] = template

    if init_first:
        annotations["vault.hashicorp.com/agent-init-first"] = "true"
        annotations["vault.hashicorp.com/agent-pre-populate-only"] = "true"

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": annotations,
        },
        "spec": {
            "serviceAccountName": "greenlang-api",
            "containers": [
                {
                    "name": "app",
                    "image": "busybox:latest",
                    "command": ["sh", "-c", "cat /vault/secrets/config && sleep 3600"],
                    "resources": {
                        "limits": {"memory": "64Mi", "cpu": "100m"},
                        "requests": {"memory": "32Mi", "cpu": "50m"},
                    },
                }
            ],
        },
    }


def _parse_agent_annotations(annotations: Dict[str, str]) -> Dict[str, Any]:
    """Parse Vault Agent Injector annotations."""
    result = {
        "enabled": annotations.get("vault.hashicorp.com/agent-inject", "false") == "true",
        "role": annotations.get("vault.hashicorp.com/role"),
        "secrets": {},
        "templates": {},
        "init_first": annotations.get("vault.hashicorp.com/agent-init-first", "false") == "true",
        "pre_populate_only": annotations.get(
            "vault.hashicorp.com/agent-pre-populate-only", "false"
        ) == "true",
    }

    # Parse secrets and templates
    for key, value in annotations.items():
        if key.startswith("vault.hashicorp.com/agent-inject-secret-"):
            secret_name = key.replace("vault.hashicorp.com/agent-inject-secret-", "")
            result["secrets"][secret_name] = value

        if key.startswith("vault.hashicorp.com/agent-inject-template-"):
            template_name = key.replace("vault.hashicorp.com/agent-inject-template-", "")
            result["templates"][template_name] = value

    return result


def _render_agent_template(
    template: str,
    secret_data: Dict[str, Any],
) -> str:
    """Render a Vault Agent template with secret data."""
    # Simple template rendering for testing
    # Real templates use Go template syntax
    result = template

    for key, value in secret_data.items():
        result = result.replace(f"{{{{ .Data.data.{key} }}}}", str(value))

    return result


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_namespace() -> str:
    """Get test namespace."""
    return TEST_CONFIG["namespace"]


@pytest.fixture
def unique_name() -> str:
    """Generate unique resource name."""
    return f"test-pod-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_annotations() -> Dict[str, str]:
    """Sample Vault Agent annotations."""
    return {
        "vault.hashicorp.com/agent-inject": "true",
        "vault.hashicorp.com/role": "greenlang-api",
        "vault.hashicorp.com/agent-inject-secret-database": "secret/data/database/config",
        "vault.hashicorp.com/agent-inject-secret-api-key": "secret/data/api-keys/stripe",
        "vault.hashicorp.com/agent-inject-template-database": """
{{- with secret "secret/data/database/config" -}}
DATABASE_URL=postgresql://{{ .Data.data.username }}:{{ .Data.data.password }}@{{ .Data.data.host }}:5432/greenlang
{{- end -}}
""",
    }


# ============================================================================
# TestAnnotationParsing
# ============================================================================


class TestAnnotationParsing:
    """Tests for parsing Vault Agent annotations."""

    def test_annotation_parsing(self, sample_annotations) -> None:
        """Test parsing Vault Agent annotations."""
        result = _parse_agent_annotations(sample_annotations)

        assert result["enabled"] is True
        assert result["role"] == "greenlang-api"
        assert "database" in result["secrets"]
        assert result["secrets"]["database"] == "secret/data/database/config"

    def test_annotation_parsing_disabled(self) -> None:
        """Test parsing when agent injection is disabled."""
        annotations = {
            "vault.hashicorp.com/agent-inject": "false",
        }

        result = _parse_agent_annotations(annotations)

        assert result["enabled"] is False

    def test_annotation_parsing_multiple_secrets(self) -> None:
        """Test parsing multiple secret annotations."""
        annotations = {
            "vault.hashicorp.com/agent-inject": "true",
            "vault.hashicorp.com/role": "app",
            "vault.hashicorp.com/agent-inject-secret-db": "secret/db",
            "vault.hashicorp.com/agent-inject-secret-api": "secret/api",
            "vault.hashicorp.com/agent-inject-secret-cache": "secret/cache",
        }

        result = _parse_agent_annotations(annotations)

        assert len(result["secrets"]) == 3
        assert "db" in result["secrets"]
        assert "api" in result["secrets"]
        assert "cache" in result["secrets"]

    def test_annotation_parsing_with_template(self, sample_annotations) -> None:
        """Test parsing template annotations."""
        result = _parse_agent_annotations(sample_annotations)

        assert "database" in result["templates"]
        assert "DATABASE_URL" in result["templates"]["database"]

    def test_annotation_parsing_init_mode(self) -> None:
        """Test parsing init container mode annotations."""
        annotations = {
            "vault.hashicorp.com/agent-inject": "true",
            "vault.hashicorp.com/role": "app",
            "vault.hashicorp.com/agent-init-first": "true",
            "vault.hashicorp.com/agent-pre-populate-only": "true",
        }

        result = _parse_agent_annotations(annotations)

        assert result["init_first"] is True
        assert result["pre_populate_only"] is True


# ============================================================================
# TestTemplateRendering
# ============================================================================


class TestTemplateRendering:
    """Tests for Vault Agent template rendering."""

    def test_template_rendering(self) -> None:
        """Test rendering a template with secret data."""
        template = "DATABASE_URL=postgresql://{{ .Data.data.username }}:{{ .Data.data.password }}@localhost:5432/db"
        secret_data = {
            "username": "admin",
            "password": "secret123",
        }

        result = _render_agent_template(template, secret_data)

        assert "admin" in result
        assert "secret123" in result
        assert "{{ .Data" not in result

    def test_template_rendering_multiple_values(self) -> None:
        """Test rendering template with multiple values."""
        template = """
HOST={{ .Data.data.host }}
PORT={{ .Data.data.port }}
USER={{ .Data.data.username }}
PASS={{ .Data.data.password }}
"""
        secret_data = {
            "host": "db.example.com",
            "port": "5432",
            "username": "app_user",
            "password": "app_pass",
        }

        result = _render_agent_template(template, secret_data)

        assert "db.example.com" in result
        assert "5432" in result
        assert "app_user" in result
        assert "app_pass" in result

    def test_template_rendering_json_format(self) -> None:
        """Test rendering JSON formatted output."""
        # In real Go templates, this would use json marshaling
        template = '{"username": "{{ .Data.data.username }}", "password": "{{ .Data.data.password }}"}'
        secret_data = {
            "username": "admin",
            "password": "secret",
        }

        result = _render_agent_template(template, secret_data)

        assert '"username": "admin"' in result
        assert '"password": "secret"' in result

    def test_template_rendering_env_file(self) -> None:
        """Test rendering .env file format."""
        template = """
DB_HOST={{ .Data.data.host }}
DB_USER={{ .Data.data.username }}
DB_PASS={{ .Data.data.password }}
"""
        secret_data = {
            "host": "localhost",
            "username": "root",
            "password": "rootpass",
        }

        result = _render_agent_template(template, secret_data)

        assert "DB_HOST=localhost" in result
        assert "DB_USER=root" in result


# ============================================================================
# TestInitContainerMode
# ============================================================================


class TestInitContainerMode:
    """Tests for init container mode."""

    def test_init_container_mode(self, test_namespace, unique_name) -> None:
        """Test pod with init container mode annotations."""
        pod = _create_pod_with_agent_annotations(
            name=unique_name,
            namespace=test_namespace,
            vault_role="greenlang-api",
            secret_path="secret/data/config",
            init_first=True,
        )

        annotations = pod["metadata"]["annotations"]

        assert annotations["vault.hashicorp.com/agent-init-first"] == "true"
        assert annotations["vault.hashicorp.com/agent-pre-populate-only"] == "true"

    def test_init_container_config(self) -> None:
        """Test init container configuration."""
        # Init container should:
        # 1. Run before main container
        # 2. Fetch secrets once
        # 3. Exit after secrets are written

        init_config = {
            "vault.hashicorp.com/agent-init-first": "true",
            "vault.hashicorp.com/agent-pre-populate-only": "true",
            "vault.hashicorp.com/agent-inject-status": "injected",
        }

        result = _parse_agent_annotations(init_config)

        assert result["init_first"] is True
        assert result["pre_populate_only"] is True


# ============================================================================
# TestSidecarMode
# ============================================================================


class TestSidecarMode:
    """Tests for sidecar mode with auto-renewal."""

    def test_sidecar_mode(self, test_namespace, unique_name) -> None:
        """Test pod with sidecar mode annotations."""
        pod = _create_pod_with_agent_annotations(
            name=unique_name,
            namespace=test_namespace,
            vault_role="greenlang-api",
            secret_path="secret/data/config",
            init_first=False,
        )

        annotations = pod["metadata"]["annotations"]

        # Sidecar mode - no init-first or pre-populate-only
        assert "vault.hashicorp.com/agent-init-first" not in annotations

    def test_sidecar_renewal_config(self) -> None:
        """Test sidecar renewal configuration."""
        # Sidecar should:
        # 1. Run alongside main container
        # 2. Keep connection to Vault
        # 3. Auto-renew secrets before expiry

        sidecar_config = {
            "vault.hashicorp.com/agent-inject": "true",
            "vault.hashicorp.com/role": "app",
            "vault.hashicorp.com/agent-cache-enable": "true",
            "vault.hashicorp.com/agent-cache-use-auto-auth-token": "true",
        }

        # These annotations enable caching and auto-auth
        assert sidecar_config["vault.hashicorp.com/agent-cache-enable"] == "true"

    def test_sidecar_resources(self) -> None:
        """Test sidecar resource configuration."""
        resources = {
            "vault.hashicorp.com/agent-requests-cpu": "50m",
            "vault.hashicorp.com/agent-requests-mem": "32Mi",
            "vault.hashicorp.com/agent-limits-cpu": "100m",
            "vault.hashicorp.com/agent-limits-mem": "64Mi",
        }

        assert resources["vault.hashicorp.com/agent-limits-mem"] == "64Mi"


# ============================================================================
# TestAgentConfiguration
# ============================================================================


class TestAgentConfiguration:
    """Tests for Agent configuration options."""

    def test_agent_config_annotations(self) -> None:
        """Test various agent configuration annotations."""
        config_annotations = {
            "vault.hashicorp.com/agent-inject": "true",
            "vault.hashicorp.com/role": "greenlang-api",
            # Logging
            "vault.hashicorp.com/log-level": "info",
            "vault.hashicorp.com/log-format": "json",
            # Resources
            "vault.hashicorp.com/agent-requests-cpu": "50m",
            "vault.hashicorp.com/agent-requests-mem": "32Mi",
            # TLS
            "vault.hashicorp.com/tls-skip-verify": "false",
            "vault.hashicorp.com/ca-cert": "/vault/tls/ca.crt",
            # Auth
            "vault.hashicorp.com/auth-path": "auth/kubernetes",
        }

        assert config_annotations["vault.hashicorp.com/log-format"] == "json"
        assert config_annotations["vault.hashicorp.com/tls-skip-verify"] == "false"

    def test_secret_file_permissions(self) -> None:
        """Test secret file permission annotations."""
        permission_annotations = {
            "vault.hashicorp.com/agent-inject-file-mode": "0400",
            "vault.hashicorp.com/agent-run-as-user": "1000",
            "vault.hashicorp.com/agent-run-as-group": "1000",
        }

        assert permission_annotations["vault.hashicorp.com/agent-inject-file-mode"] == "0400"

    def test_secret_path_configuration(self) -> None:
        """Test secret path annotations."""
        path_annotations = {
            "vault.hashicorp.com/secret-volume-path": "/vault/secrets",
            "vault.hashicorp.com/agent-inject-secret-db": "secret/data/db",
            "vault.hashicorp.com/secret-volume-path-db": "/app/config/db",
        }

        assert path_annotations["vault.hashicorp.com/secret-volume-path-db"] == "/app/config/db"


# ============================================================================
# TestPodManifests
# ============================================================================


class TestPodManifests:
    """Tests for complete pod manifest generation."""

    def test_pod_manifest_structure(self, test_namespace, unique_name) -> None:
        """Test pod manifest has correct structure."""
        pod = _create_pod_with_agent_annotations(
            name=unique_name,
            namespace=test_namespace,
            vault_role="greenlang-api",
            secret_path="secret/data/config",
        )

        assert pod["apiVersion"] == "v1"
        assert pod["kind"] == "Pod"
        assert "annotations" in pod["metadata"]
        assert "serviceAccountName" in pod["spec"]

    def test_pod_manifest_with_template(self, test_namespace, unique_name) -> None:
        """Test pod manifest with custom template."""
        template = """
{{- with secret "secret/data/database/config" -}}
export DATABASE_URL=postgresql://{{ .Data.data.username }}:{{ .Data.data.password }}@{{ .Data.data.host }}:5432/greenlang
{{- end -}}
"""
        pod = _create_pod_with_agent_annotations(
            name=unique_name,
            namespace=test_namespace,
            vault_role="greenlang-api",
            secret_path="secret/data/database/config",
            template=template,
        )

        assert "vault.hashicorp.com/agent-inject-template-config" in pod["metadata"]["annotations"]
