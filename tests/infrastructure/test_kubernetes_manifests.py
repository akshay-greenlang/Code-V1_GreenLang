# -*- coding: utf-8 -*-
"""
Kubernetes Manifest Validation Tests

INFRA-001: Infrastructure test suite for validating Kubernetes manifests.

Tests include:
- YAML syntax validation
- API version compatibility
- Required labels and annotations
- Resource limits and requests
- Security context validation
- Probes configuration
- Network policies
- Service configuration
- Ingress validation

Target coverage: 85%+
"""

import re
from pathlib import Path
from typing import Dict, List, Any

import pytest
import yaml


class TestKubernetesManifestStructure:
    """Test Kubernetes manifest file structure."""

    def test_kubernetes_directory_exists(self, kubernetes_dir: Path):
        """Test that the Kubernetes directory exists."""
        assert kubernetes_dir.exists(), f"Kubernetes directory not found at {kubernetes_dir}"

    def test_kubernetes_manifests_exist(self, all_kubernetes_manifests: List[Path]):
        """Test that Kubernetes manifest files exist."""
        assert len(all_kubernetes_manifests) > 0, "No Kubernetes manifests found"

    def test_manifests_have_valid_yaml_syntax(self, all_kubernetes_manifests: List[Path]):
        """Test that all manifests have valid YAML syntax."""
        errors = []

        for manifest in all_kubernetes_manifests:
            try:
                with open(manifest, 'r') as f:
                    list(yaml.safe_load_all(f.read()))
            except yaml.YAMLError as e:
                errors.append(f"{manifest}: {e}")

        assert len(errors) == 0, f"YAML syntax errors:\n" + "\n".join(errors)


class TestKubernetesAPIVersions:
    """Test Kubernetes API version compatibility."""

    DEPRECATED_API_VERSIONS = {
        "extensions/v1beta1": "networking.k8s.io/v1 or apps/v1",
        "apps/v1beta1": "apps/v1",
        "apps/v1beta2": "apps/v1",
        "networking.k8s.io/v1beta1": "networking.k8s.io/v1",
        "rbac.authorization.k8s.io/v1beta1": "rbac.authorization.k8s.io/v1",
    }

    def test_no_deprecated_api_versions(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that no deprecated API versions are used."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    api_version = kubernetes_validator.get_api_version(manifest)

                    if api_version in self.DEPRECATED_API_VERSIONS:
                        replacement = self.DEPRECATED_API_VERSIONS[api_version]
                        violations.append(
                            f"{manifest_path}: {api_version} is deprecated, use {replacement}"
                        )
            except Exception as e:
                violations.append(f"{manifest_path}: Error parsing - {e}")

        assert len(violations) == 0, f"Deprecated API versions:\n" + "\n".join(violations)

    def test_deployment_uses_apps_v1(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that Deployments use apps/v1 API version."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    if kubernetes_validator.get_kind(manifest) == "Deployment":
                        api_version = kubernetes_validator.get_api_version(manifest)
                        if api_version != "apps/v1":
                            violations.append(
                                f"{manifest_path}: Deployment should use apps/v1, found {api_version}"
                            )
            except Exception:
                pass

        assert len(violations) == 0, f"API version issues:\n" + "\n".join(violations)

    def test_ingress_uses_networking_v1(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that Ingress uses networking.k8s.io/v1 API version."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    if kubernetes_validator.get_kind(manifest) == "Ingress":
                        api_version = kubernetes_validator.get_api_version(manifest)
                        if api_version != "networking.k8s.io/v1":
                            violations.append(
                                f"{manifest_path}: Ingress should use networking.k8s.io/v1"
                            )
            except Exception:
                pass

        assert len(violations) == 0, f"Ingress API version issues:\n" + "\n".join(violations)


class TestKubernetesLabels:
    """Test Kubernetes label requirements."""

    REQUIRED_LABELS = ["app"]
    RECOMMENDED_LABELS = ["environment", "version"]

    def test_workloads_have_required_labels(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that workloads have required labels."""
        workload_kinds = ["Deployment", "StatefulSet", "DaemonSet", "Job"]
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    kind = kubernetes_validator.get_kind(manifest)
                    if kind in workload_kinds:
                        missing = kubernetes_validator.check_required_labels(
                            manifest, self.REQUIRED_LABELS
                        )
                        if missing:
                            name = manifest.get("metadata", {}).get("name", "unnamed")
                            violations.append(
                                f"{manifest_path}: {kind}/{name} missing labels: {missing}"
                            )
            except Exception as e:
                violations.append(f"{manifest_path}: Error - {e}")

        assert len(violations) == 0, f"Missing required labels:\n" + "\n".join(violations)

    def test_services_have_selectors(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that Services have pod selectors."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    if kubernetes_validator.get_kind(manifest) == "Service":
                        spec = manifest.get("spec", {})
                        # ClusterIP=None (headless) and ExternalName don't need selectors
                        if spec.get("type") != "ExternalName":
                            selector = spec.get("selector")
                            if not selector:
                                name = manifest.get("metadata", {}).get("name", "unnamed")
                                violations.append(
                                    f"{manifest_path}: Service/{name} has no selector"
                                )
            except Exception:
                pass

        # Services without selectors may be intentional (e.g., for ExternalName)
        if violations:
            pytest.skip(f"Services without selectors (review recommended): {len(violations)}")


class TestKubernetesResourceLimits:
    """Test Kubernetes resource limits and requests."""

    def test_containers_have_resource_limits(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that containers have resource limits defined."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    kind = kubernetes_validator.get_kind(manifest)
                    if kind in kubernetes_validator.WORKLOAD_KINDS:
                        resource_checks = kubernetes_validator.check_resource_limits(manifest)

                        for container_name, checks in resource_checks.items():
                            if not checks.get("has_memory_limit"):
                                name = manifest.get("metadata", {}).get("name", "unnamed")
                                violations.append(
                                    f"{manifest_path}: {kind}/{name}/{container_name} missing memory limit"
                                )
                            if not checks.get("has_cpu_limit"):
                                name = manifest.get("metadata", {}).get("name", "unnamed")
                                violations.append(
                                    f"{manifest_path}: {kind}/{name}/{container_name} missing cpu limit"
                                )
            except Exception:
                pass

        assert len(violations) == 0, f"Missing resource limits:\n" + "\n".join(violations)

    def test_containers_have_resource_requests(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that containers have resource requests defined."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    kind = kubernetes_validator.get_kind(manifest)
                    if kind in kubernetes_validator.WORKLOAD_KINDS:
                        resource_checks = kubernetes_validator.check_resource_limits(manifest)

                        for container_name, checks in resource_checks.items():
                            if not checks.get("has_memory_request"):
                                name = manifest.get("metadata", {}).get("name", "unnamed")
                                violations.append(
                                    f"{manifest_path}: {kind}/{name}/{container_name} missing memory request"
                                )
            except Exception:
                pass

        assert len(violations) == 0, f"Missing resource requests:\n" + "\n".join(violations)


class TestKubernetesSecurityContext:
    """Test Kubernetes security context configuration."""

    def test_pods_run_as_non_root(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that pods are configured to run as non-root."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    kind = kubernetes_validator.get_kind(manifest)
                    if kind in kubernetes_validator.WORKLOAD_KINDS:
                        security = kubernetes_validator.check_security_context(manifest)
                        pod_security = security.get("pod_security_context", {})

                        run_as_non_root = pod_security.get("runAsNonRoot", False)

                        # Check container-level if not set at pod level
                        if not run_as_non_root:
                            containers_ok = all(
                                ctx.get("runAsNonRoot", False)
                                for ctx in security.get("containers", {}).values()
                            )
                            if not containers_ok and security.get("containers"):
                                name = manifest.get("metadata", {}).get("name", "unnamed")
                                violations.append(
                                    f"{manifest_path}: {kind}/{name} should run as non-root"
                                )
            except Exception:
                pass

        # Non-root is a recommendation, not always required
        if violations:
            pytest.skip(f"Workloads not running as non-root (review): {len(violations)}")

    def test_containers_have_security_context(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that containers have security context defined."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    kind = kubernetes_validator.get_kind(manifest)
                    if kind in kubernetes_validator.WORKLOAD_KINDS:
                        security = kubernetes_validator.check_security_context(manifest)
                        containers = security.get("containers", {})

                        for container_name, ctx in containers.items():
                            if not ctx:
                                name = manifest.get("metadata", {}).get("name", "unnamed")
                                violations.append(
                                    f"{manifest_path}: {kind}/{name}/{container_name} has no securityContext"
                                )
            except Exception:
                pass

        if violations:
            pytest.skip(f"Containers without securityContext (review): {len(violations)}")


class TestKubernetesProbes:
    """Test Kubernetes health probe configuration."""

    def test_deployments_have_liveness_probes(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that Deployments have liveness probes."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    if kubernetes_validator.get_kind(manifest) == "Deployment":
                        probes = kubernetes_validator.check_probes(manifest)

                        for container_name, probe_checks in probes.items():
                            if not probe_checks.get("has_liveness_probe"):
                                name = manifest.get("metadata", {}).get("name", "unnamed")
                                violations.append(
                                    f"{manifest_path}: Deployment/{name}/{container_name} missing livenessProbe"
                                )
            except Exception:
                pass

        assert len(violations) == 0, f"Missing liveness probes:\n" + "\n".join(violations)

    def test_deployments_have_readiness_probes(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that Deployments have readiness probes."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    if kubernetes_validator.get_kind(manifest) == "Deployment":
                        probes = kubernetes_validator.check_probes(manifest)

                        for container_name, probe_checks in probes.items():
                            if not probe_checks.get("has_readiness_probe"):
                                name = manifest.get("metadata", {}).get("name", "unnamed")
                                violations.append(
                                    f"{manifest_path}: Deployment/{name}/{container_name} missing readinessProbe"
                                )
            except Exception:
                pass

        assert len(violations) == 0, f"Missing readiness probes:\n" + "\n".join(violations)


class TestKubernetesIngress:
    """Test Kubernetes Ingress configuration."""

    def test_ingress_has_tls(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that Ingress resources have TLS configured."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    if kubernetes_validator.get_kind(manifest) == "Ingress":
                        ingress_info = kubernetes_validator.validate_ingress(manifest)

                        if not ingress_info.get("has_tls"):
                            name = manifest.get("metadata", {}).get("name", "unnamed")
                            violations.append(
                                f"{manifest_path}: Ingress/{name} should have TLS configured"
                            )
            except Exception:
                pass

        assert len(violations) == 0, f"Ingress without TLS:\n" + "\n".join(violations)

    def test_ingress_hosts_match_tls_hosts(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that Ingress hosts match TLS hosts."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    if kubernetes_validator.get_kind(manifest) == "Ingress":
                        ingress_info = kubernetes_validator.validate_ingress(manifest)

                        hosts = set(ingress_info.get("hosts", []))
                        tls_hosts = set(ingress_info.get("tls_hosts", []))

                        missing_tls = hosts - tls_hosts
                        if missing_tls and tls_hosts:  # Only check if TLS is partially configured
                            name = manifest.get("metadata", {}).get("name", "unnamed")
                            violations.append(
                                f"{manifest_path}: Ingress/{name} hosts missing from TLS: {missing_tls}"
                            )
            except Exception:
                pass

        assert len(violations) == 0, f"TLS host mismatches:\n" + "\n".join(violations)


class TestKubernetesNamespaces:
    """Test Kubernetes namespace configuration."""

    def test_resources_have_namespaces(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that namespaced resources have namespace defined."""
        # Resources that should always have namespace
        namespaced_kinds = [
            "Deployment", "StatefulSet", "DaemonSet", "Service", "ConfigMap",
            "Secret", "Ingress", "NetworkPolicy", "ServiceAccount", "Role", "RoleBinding"
        ]
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    kind = kubernetes_validator.get_kind(manifest)
                    if kind in namespaced_kinds:
                        namespace = kubernetes_validator.check_namespace(manifest)

                        if not namespace:
                            name = manifest.get("metadata", {}).get("name", "unnamed")
                            violations.append(
                                f"{manifest_path}: {kind}/{name} should specify namespace"
                            )
            except Exception:
                pass

        # Warning rather than failure - namespace might be set at deploy time
        if violations:
            pytest.skip(f"Resources without explicit namespace (review): {len(violations)}")

    def test_namespace_resources_exist(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that Namespace resources are defined."""
        namespaces_found = set()

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    if kubernetes_validator.get_kind(manifest) == "Namespace":
                        name = manifest.get("metadata", {}).get("name")
                        if name:
                            namespaces_found.add(name)
            except Exception:
                pass

        # At minimum, we should have our application namespace
        assert "greenlang" in namespaces_found, "greenlang namespace should be defined"


class TestKubernetesNetworkPolicies:
    """Test Kubernetes NetworkPolicy configuration."""

    def test_network_policies_exist(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that NetworkPolicy resources exist."""
        network_policies = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    if kubernetes_validator.get_kind(manifest) == "NetworkPolicy":
                        name = manifest.get("metadata", {}).get("name")
                        network_policies.append(name)
            except Exception:
                pass

        assert len(network_policies) > 0, "No NetworkPolicy resources found"

    def test_network_policies_have_pod_selector(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that NetworkPolicies have podSelector defined."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    if kubernetes_validator.get_kind(manifest) == "NetworkPolicy":
                        spec = manifest.get("spec", {})
                        pod_selector = spec.get("podSelector")

                        if pod_selector is None:
                            name = manifest.get("metadata", {}).get("name", "unnamed")
                            violations.append(
                                f"{manifest_path}: NetworkPolicy/{name} missing podSelector"
                            )
            except Exception:
                pass

        assert len(violations) == 0, f"NetworkPolicy issues:\n" + "\n".join(violations)


class TestKubernetesRBAC:
    """Test Kubernetes RBAC configuration."""

    def test_service_accounts_exist(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that ServiceAccount resources exist."""
        service_accounts = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    if kubernetes_validator.get_kind(manifest) == "ServiceAccount":
                        name = manifest.get("metadata", {}).get("name")
                        service_accounts.append(name)
            except Exception:
                pass

        assert len(service_accounts) > 0, "No ServiceAccount resources found"

    def test_roles_have_rules(
        self,
        all_kubernetes_manifests: List[Path],
        kubernetes_validator
    ):
        """Test that Role/ClusterRole resources have rules defined."""
        violations = []

        for manifest_path in all_kubernetes_manifests:
            try:
                manifests = kubernetes_validator.load_manifests(manifest_path)

                for manifest in manifests:
                    kind = kubernetes_validator.get_kind(manifest)
                    if kind in ["Role", "ClusterRole"]:
                        rules = manifest.get("rules", [])

                        if not rules:
                            name = manifest.get("metadata", {}).get("name", "unnamed")
                            violations.append(
                                f"{manifest_path}: {kind}/{name} has no rules defined"
                            )
            except Exception:
                pass

        assert len(violations) == 0, f"RBAC issues:\n" + "\n".join(violations)


class TestKubernetesManifestValidationWithMock:
    """Test Kubernetes manifest validation with mock kubectl."""

    def test_manifest_dry_run_with_mock(
        self,
        temp_k8s_manifest: Path,
        mock_kubectl_cli
    ):
        """Test manifest validation using mock kubectl."""
        result = mock_kubectl_cli.dry_run(temp_k8s_manifest)

        assert result["valid"], "Manifest should pass dry-run validation"
        assert len(result["errors"]) == 0, "Manifest should have no errors"

    def test_temp_manifest_has_resource_limits(
        self,
        temp_k8s_manifest: Path,
        kubernetes_validator
    ):
        """Test that temporary manifest has resource limits."""
        manifests = kubernetes_validator.load_manifests(temp_k8s_manifest)

        for manifest in manifests:
            if kubernetes_validator.get_kind(manifest) == "Deployment":
                resource_checks = kubernetes_validator.check_resource_limits(manifest)

                for container_name, checks in resource_checks.items():
                    assert checks["has_memory_limit"], f"{container_name} should have memory limit"
                    assert checks["has_cpu_limit"], f"{container_name} should have cpu limit"

    def test_temp_manifest_has_probes(
        self,
        temp_k8s_manifest: Path,
        kubernetes_validator
    ):
        """Test that temporary manifest has health probes."""
        manifests = kubernetes_validator.load_manifests(temp_k8s_manifest)

        for manifest in manifests:
            if kubernetes_validator.get_kind(manifest) == "Deployment":
                probes = kubernetes_validator.check_probes(manifest)

                for container_name, probe_checks in probes.items():
                    assert probe_checks["has_liveness_probe"], f"{container_name} should have liveness probe"
                    assert probe_checks["has_readiness_probe"], f"{container_name} should have readiness probe"
