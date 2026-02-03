# -*- coding: utf-8 -*-
"""
Pytest fixtures for infrastructure testing.

Provides fixtures for Terraform, Kubernetes, and Helm validation.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock

import pytest
import yaml


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def terraform_dir(project_root) -> Path:
    """Get the Terraform configuration directory."""
    return project_root / "deployment" / "terraform"


@pytest.fixture(scope="session")
def kubernetes_dir(project_root) -> Path:
    """Get the Kubernetes manifests directory."""
    return project_root / "deployment" / "kubernetes"


@pytest.fixture(scope="session")
def helm_dir(project_root) -> Path:
    """Get the Helm charts directory."""
    return project_root / "deployment" / "helm"


# =============================================================================
# Terraform Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def terraform_modules(terraform_dir) -> List[Path]:
    """List all Terraform modules."""
    modules_dir = terraform_dir / "modules"
    if not modules_dir.exists():
        return []
    return [d for d in modules_dir.iterdir() if d.is_dir()]


@pytest.fixture(scope="session")
def terraform_environments(terraform_dir) -> List[Path]:
    """List all Terraform environments."""
    envs_dir = terraform_dir / "environments"
    if not envs_dir.exists():
        return []
    return [d for d in envs_dir.iterdir() if d.is_dir()]


@pytest.fixture(scope="session")
def all_terraform_files(terraform_dir) -> List[Path]:
    """Get all Terraform files in the project."""
    if not terraform_dir.exists():
        return []
    return list(terraform_dir.rglob("*.tf"))


@pytest.fixture
def terraform_validator():
    """Terraform file validator utilities."""

    class TerraformValidator:
        """Utilities for validating Terraform configurations."""

        # Required provider versions
        REQUIRED_PROVIDERS = {
            "aws": ">= 5.0",
            "kubernetes": ">= 2.0",
            "random": ">= 3.0",
        }

        # Security-sensitive resources that need encryption
        ENCRYPTION_REQUIRED_RESOURCES = [
            "aws_rds_cluster",
            "aws_db_instance",
            "aws_elasticache_replication_group",
            "aws_s3_bucket",
            "aws_ebs_volume",
            "aws_efs_file_system",
        ]

        # Resources that should have tags
        TAGGABLE_RESOURCES = [
            "aws_vpc",
            "aws_subnet",
            "aws_security_group",
            "aws_instance",
            "aws_eks_cluster",
            "aws_rds_cluster",
            "aws_db_instance",
            "aws_elasticache_replication_group",
            "aws_s3_bucket",
        ]

        @staticmethod
        def parse_hcl_basic(content: str) -> Dict[str, Any]:
            """
            Basic HCL parser for validation purposes.

            Note: For production use, consider using python-hcl2 library.
            """
            result = {
                "terraform": [],
                "provider": [],
                "resource": [],
                "module": [],
                "variable": [],
                "output": [],
                "data": [],
                "locals": [],
            }

            lines = content.split('\n')
            current_block = None
            brace_count = 0
            block_content = []

            for line in lines:
                stripped = line.strip()

                # Detect block starts
                for block_type in result.keys():
                    if stripped.startswith(f"{block_type} ") or stripped.startswith(f"{block_type} {{"):
                        current_block = block_type
                        brace_count = 0
                        block_content = [line]
                        break

                if current_block:
                    if '{' in stripped:
                        brace_count += stripped.count('{')
                    if '}' in stripped:
                        brace_count -= stripped.count('}')

                    if current_block and line not in block_content:
                        block_content.append(line)

                    if brace_count <= 0 and '}' in stripped:
                        result[current_block].append('\n'.join(block_content))
                        current_block = None
                        block_content = []

            return result

        @staticmethod
        def check_required_files(module_path: Path) -> Dict[str, bool]:
            """Check for required Terraform files in a module."""
            required = ["main.tf", "variables.tf", "outputs.tf"]
            return {
                f: (module_path / f).exists() for f in required
            }

        @staticmethod
        def extract_resource_types(content: str) -> List[str]:
            """Extract resource types from Terraform content."""
            import re
            pattern = r'resource\s+"([^"]+)"'
            return re.findall(pattern, content)

        @staticmethod
        def check_encryption_settings(content: str, resource_type: str) -> bool:
            """Check if encryption is enabled for a resource."""
            encryption_patterns = [
                r'storage_encrypted\s*=\s*true',
                r'encrypted\s*=\s*true',
                r'at_rest_encryption_enabled\s*=\s*true',
                r'transit_encryption_enabled\s*=\s*true',
                r'kms_key_id\s*=',
                r'server_side_encryption_configuration',
            ]

            import re
            for pattern in encryption_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
            return False

        @staticmethod
        def check_tags_present(content: str) -> bool:
            """Check if tags are present in resource definitions."""
            import re
            return bool(re.search(r'\btags\s*=', content))

        @staticmethod
        def validate_variable_descriptions(content: str) -> List[str]:
            """Validate that all variables have descriptions."""
            import re
            variables = re.findall(r'variable\s+"([^"]+)"', content)
            missing_descriptions = []

            for var in variables:
                var_pattern = rf'variable\s+"{var}"\s*\{{[^}}]*description\s*='
                if not re.search(var_pattern, content, re.DOTALL):
                    missing_descriptions.append(var)

            return missing_descriptions

    return TerraformValidator()


# =============================================================================
# Kubernetes Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def all_kubernetes_manifests(kubernetes_dir) -> List[Path]:
    """Get all Kubernetes YAML manifests."""
    if not kubernetes_dir.exists():
        return []
    return list(kubernetes_dir.rglob("*.yaml")) + list(kubernetes_dir.rglob("*.yml"))


@pytest.fixture
def kubernetes_validator():
    """Kubernetes manifest validator utilities."""

    class KubernetesValidator:
        """Utilities for validating Kubernetes manifests."""

        # Required labels for production resources
        REQUIRED_LABELS = ["app", "environment"]

        # Resource types that need resource limits
        WORKLOAD_KINDS = ["Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"]

        # Security-sensitive settings
        SECURITY_CHECKS = {
            "runAsNonRoot": True,
            "readOnlyRootFilesystem": True,
            "allowPrivilegeEscalation": False,
        }

        @staticmethod
        def load_manifests(file_path: Path) -> List[Dict[str, Any]]:
            """Load all YAML documents from a manifest file."""
            with open(file_path, 'r') as f:
                content = f.read()

            documents = []
            for doc in yaml.safe_load_all(content):
                if doc is not None:
                    documents.append(doc)
            return documents

        @staticmethod
        def get_kind(manifest: Dict[str, Any]) -> str:
            """Get the kind of a Kubernetes manifest."""
            return manifest.get("kind", "Unknown")

        @staticmethod
        def get_api_version(manifest: Dict[str, Any]) -> str:
            """Get the API version of a Kubernetes manifest."""
            return manifest.get("apiVersion", "Unknown")

        @staticmethod
        def check_required_labels(manifest: Dict[str, Any], required: List[str]) -> List[str]:
            """Check for required labels in manifest metadata."""
            labels = manifest.get("metadata", {}).get("labels", {})
            missing = [label for label in required if label not in labels]
            return missing

        @staticmethod
        def check_resource_limits(manifest: Dict[str, Any]) -> Dict[str, bool]:
            """Check if resource limits are set for containers."""
            results = {}
            kind = manifest.get("kind", "")

            if kind not in ["Deployment", "StatefulSet", "DaemonSet", "Job"]:
                return results

            # Navigate to container specs
            spec = manifest.get("spec", {})
            template = spec.get("template", {})
            pod_spec = template.get("spec", {})
            containers = pod_spec.get("containers", [])

            for container in containers:
                name = container.get("name", "unnamed")
                resources = container.get("resources", {})
                limits = resources.get("limits", {})
                requests = resources.get("requests", {})

                results[name] = {
                    "has_cpu_limit": "cpu" in limits,
                    "has_memory_limit": "memory" in limits,
                    "has_cpu_request": "cpu" in requests,
                    "has_memory_request": "memory" in requests,
                }

            return results

        @staticmethod
        def check_security_context(manifest: Dict[str, Any]) -> Dict[str, Any]:
            """Check security context settings for pods/containers."""
            kind = manifest.get("kind", "")

            if kind not in ["Deployment", "StatefulSet", "DaemonSet", "Job", "Pod"]:
                return {}

            if kind == "Pod":
                pod_spec = manifest.get("spec", {})
            else:
                spec = manifest.get("spec", {})
                template = spec.get("template", {})
                pod_spec = template.get("spec", {})

            pod_security = pod_spec.get("securityContext", {})

            results = {
                "pod_security_context": pod_security,
                "containers": {}
            }

            for container in pod_spec.get("containers", []):
                name = container.get("name", "unnamed")
                container_security = container.get("securityContext", {})
                results["containers"][name] = container_security

            return results

        @staticmethod
        def check_probes(manifest: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
            """Check if health probes are configured for containers."""
            results = {}
            kind = manifest.get("kind", "")

            if kind not in ["Deployment", "StatefulSet", "DaemonSet"]:
                return results

            spec = manifest.get("spec", {})
            template = spec.get("template", {})
            pod_spec = template.get("spec", {})
            containers = pod_spec.get("containers", [])

            for container in containers:
                name = container.get("name", "unnamed")
                results[name] = {
                    "has_liveness_probe": "livenessProbe" in container,
                    "has_readiness_probe": "readinessProbe" in container,
                    "has_startup_probe": "startupProbe" in container,
                }

            return results

        @staticmethod
        def validate_ingress(manifest: Dict[str, Any]) -> Dict[str, Any]:
            """Validate Ingress configuration."""
            if manifest.get("kind") != "Ingress":
                return {}

            spec = manifest.get("spec", {})

            return {
                "has_tls": bool(spec.get("tls")),
                "rules_count": len(spec.get("rules", [])),
                "hosts": [
                    rule.get("host")
                    for rule in spec.get("rules", [])
                    if rule.get("host")
                ],
                "tls_hosts": [
                    host
                    for tls in spec.get("tls", [])
                    for host in tls.get("hosts", [])
                ],
            }

        @staticmethod
        def check_namespace(manifest: Dict[str, Any]) -> Optional[str]:
            """Check the namespace of a manifest."""
            return manifest.get("metadata", {}).get("namespace")

    return KubernetesValidator()


# =============================================================================
# Helm Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def all_helm_charts(helm_dir) -> List[Path]:
    """Get all Helm chart directories."""
    if not helm_dir.exists():
        return []

    charts = []
    for item in helm_dir.rglob("Chart.yaml"):
        charts.append(item.parent)
    return charts


@pytest.fixture
def helm_validator():
    """Helm chart validator utilities."""

    class HelmValidator:
        """Utilities for validating Helm charts."""

        # Required files in a Helm chart
        REQUIRED_FILES = ["Chart.yaml", "values.yaml"]

        # Required fields in Chart.yaml
        REQUIRED_CHART_FIELDS = ["apiVersion", "name", "version"]

        # Recommended fields in Chart.yaml
        RECOMMENDED_CHART_FIELDS = ["description", "type", "appVersion", "maintainers"]

        @staticmethod
        def load_chart_yaml(chart_path: Path) -> Dict[str, Any]:
            """Load Chart.yaml from a Helm chart."""
            chart_file = chart_path / "Chart.yaml"
            if not chart_file.exists():
                return {}

            with open(chart_file, 'r') as f:
                return yaml.safe_load(f) or {}

        @staticmethod
        def load_values_yaml(chart_path: Path, values_file: str = "values.yaml") -> Dict[str, Any]:
            """Load values.yaml or a variant from a Helm chart."""
            values_path = chart_path / values_file
            if not values_path.exists():
                return {}

            with open(values_path, 'r') as f:
                return yaml.safe_load(f) or {}

        @staticmethod
        def check_required_files(chart_path: Path) -> Dict[str, bool]:
            """Check for required files in a Helm chart."""
            required = ["Chart.yaml", "values.yaml", "templates"]
            return {
                item: (chart_path / item).exists() for item in required
            }

        @staticmethod
        def check_chart_fields(chart_yaml: Dict[str, Any], required: List[str]) -> List[str]:
            """Check for required fields in Chart.yaml."""
            return [field for field in required if field not in chart_yaml]

        @staticmethod
        def get_dependencies(chart_yaml: Dict[str, Any]) -> List[Dict[str, str]]:
            """Get chart dependencies."""
            return chart_yaml.get("dependencies", [])

        @staticmethod
        def check_templates_exist(chart_path: Path) -> List[str]:
            """Get list of templates in the chart."""
            templates_dir = chart_path / "templates"
            if not templates_dir.exists():
                return []

            return [
                f.name for f in templates_dir.iterdir()
                if f.is_file() and f.suffix in [".yaml", ".yml", ".tpl"]
            ]

        @staticmethod
        def validate_values_schema(values: Dict[str, Any]) -> Dict[str, Any]:
            """Validate values against common patterns."""
            issues = {
                "missing_defaults": [],
                "security_concerns": [],
                "recommendations": [],
            }

            # Check for common security settings
            if "securityContext" not in values:
                issues["recommendations"].append("Consider adding default securityContext")

            if "resources" not in values:
                issues["recommendations"].append("Consider adding default resource limits")

            # Check replica count
            replica_count = values.get("replicaCount", values.get("replicas", 1))
            if replica_count == 1:
                issues["recommendations"].append("Single replica may not provide HA")

            return issues

        @staticmethod
        def list_values_files(chart_path: Path) -> List[str]:
            """List all values files in a chart."""
            return [
                f.name for f in chart_path.iterdir()
                if f.is_file() and f.name.startswith("values") and f.suffix in [".yaml", ".yml"]
            ]

    return HelmValidator()


# =============================================================================
# Mock Fixtures for CLI Tools
# =============================================================================

@pytest.fixture
def mock_terraform_cli():
    """Mock Terraform CLI for testing without actual binary."""
    class MockTerraformCLI:
        def __init__(self):
            self.commands_run = []

        def validate(self, path: Path) -> Dict[str, Any]:
            """Mock terraform validate."""
            self.commands_run.append(f"terraform validate {path}")
            return {"valid": True, "error_count": 0, "warning_count": 0}

        def fmt_check(self, path: Path) -> Dict[str, Any]:
            """Mock terraform fmt -check."""
            self.commands_run.append(f"terraform fmt -check {path}")
            return {"formatted": True, "files": []}

        def plan(self, path: Path) -> Dict[str, Any]:
            """Mock terraform plan."""
            self.commands_run.append(f"terraform plan {path}")
            return {"changes": {"add": 0, "change": 0, "destroy": 0}}

    return MockTerraformCLI()


@pytest.fixture
def mock_kubectl_cli():
    """Mock kubectl CLI for testing without actual binary."""
    class MockKubectlCLI:
        def __init__(self):
            self.commands_run = []

        def dry_run(self, manifest_path: Path) -> Dict[str, Any]:
            """Mock kubectl apply --dry-run."""
            self.commands_run.append(f"kubectl apply --dry-run=client -f {manifest_path}")
            return {"valid": True, "errors": []}

        def get_api_resources(self) -> List[Dict[str, str]]:
            """Mock kubectl api-resources."""
            return [
                {"name": "deployments", "apiVersion": "apps/v1"},
                {"name": "services", "apiVersion": "v1"},
                {"name": "ingresses", "apiVersion": "networking.k8s.io/v1"},
            ]

    return MockKubectlCLI()


@pytest.fixture
def mock_helm_cli():
    """Mock Helm CLI for testing without actual binary."""
    class MockHelmCLI:
        def __init__(self):
            self.commands_run = []

        def lint(self, chart_path: Path) -> Dict[str, Any]:
            """Mock helm lint."""
            self.commands_run.append(f"helm lint {chart_path}")
            return {"passed": True, "messages": []}

        def template(self, chart_path: Path, values_file: Optional[str] = None) -> str:
            """Mock helm template."""
            cmd = f"helm template {chart_path}"
            if values_file:
                cmd += f" -f {values_file}"
            self.commands_run.append(cmd)
            return "# Mocked Helm template output"

        def dependency_build(self, chart_path: Path) -> Dict[str, Any]:
            """Mock helm dependency build."""
            self.commands_run.append(f"helm dependency build {chart_path}")
            return {"success": True}

    return MockHelmCLI()


# =============================================================================
# Test Environment Fixtures
# =============================================================================

@pytest.fixture
def infrastructure_test_env():
    """Set up test environment variables for infrastructure testing."""
    original_env = os.environ.copy()

    # Set test environment variables
    test_env = {
        "TF_VAR_environment": "test",
        "TF_VAR_aws_region": "us-east-1",
        "KUBECONFIG": "/tmp/test-kubeconfig",
        "HELM_DEBUG": "true",
    }

    os.environ.update(test_env)

    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_terraform_module(tmp_path):
    """Create a temporary Terraform module for testing."""
    module_dir = tmp_path / "test-module"
    module_dir.mkdir()

    # Create main.tf
    main_tf = module_dir / "main.tf"
    main_tf.write_text('''
terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

resource "aws_s3_bucket" "test" {
  bucket = var.bucket_name

  tags = var.tags
}

resource "aws_s3_bucket_server_side_encryption_configuration" "test" {
  bucket = aws_s3_bucket.test.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
''')

    # Create variables.tf
    variables_tf = module_dir / "variables.tf"
    variables_tf.write_text('''
variable "bucket_name" {
  description = "Name of the S3 bucket"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
''')

    # Create outputs.tf
    outputs_tf = module_dir / "outputs.tf"
    outputs_tf.write_text('''
output "bucket_id" {
  description = "ID of the S3 bucket"
  value       = aws_s3_bucket.test.id
}

output "bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.test.arn
}
''')

    return module_dir


@pytest.fixture
def temp_k8s_manifest(tmp_path):
    """Create a temporary Kubernetes manifest for testing."""
    manifest_file = tmp_path / "deployment.yaml"
    manifest_content = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-app
  namespace: default
  labels:
    app: test-app
    environment: test
spec:
  replicas: 2
  selector:
    matchLabels:
      app: test-app
  template:
    metadata:
      labels:
        app: test-app
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
        - name: app
          image: nginx:1.21
          ports:
            - containerPort: 80
          resources:
            limits:
              cpu: "500m"
              memory: "256Mi"
            requests:
              cpu: "100m"
              memory: "128Mi"
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
          livenessProbe:
            httpGet:
              path: /healthz
              port: 80
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 80
            initialDelaySeconds: 5
            periodSeconds: 5
'''
    manifest_file.write_text(manifest_content)
    return manifest_file


@pytest.fixture
def temp_helm_chart(tmp_path):
    """Create a temporary Helm chart for testing."""
    chart_dir = tmp_path / "test-chart"
    chart_dir.mkdir()
    templates_dir = chart_dir / "templates"
    templates_dir.mkdir()

    # Create Chart.yaml
    chart_yaml = chart_dir / "Chart.yaml"
    chart_yaml.write_text('''
apiVersion: v2
name: test-chart
description: A test Helm chart
type: application
version: 1.0.0
appVersion: "1.0.0"

maintainers:
  - name: Test Team
    email: test@example.com

keywords:
  - test
''')

    # Create values.yaml
    values_yaml = chart_dir / "values.yaml"
    values_yaml.write_text('''
replicaCount: 2

image:
  repository: nginx
  tag: "1.21"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

resources:
  limits:
    cpu: 500m
    memory: 256Mi
  requests:
    cpu: 100m
    memory: 128Mi

securityContext:
  runAsNonRoot: true
  runAsUser: 1000

nodeSelector: {}

tolerations: []

affinity: {}
''')

    # Create a basic deployment template
    deployment_tpl = templates_dir / "deployment.yaml"
    deployment_tpl.write_text('''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}
  labels:
    app: {{ .Release.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ .Release.Name }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          ports:
            - containerPort: {{ .Values.service.port }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
''')

    # Create _helpers.tpl
    helpers_tpl = templates_dir / "_helpers.tpl"
    helpers_tpl.write_text('''
{{/*
Common labels
*/}}
{{- define "test-chart.labels" -}}
app: {{ .Release.Name }}
chart: {{ .Chart.Name }}-{{ .Chart.Version }}
release: {{ .Release.Name }}
{{- end }}
''')

    return chart_dir
