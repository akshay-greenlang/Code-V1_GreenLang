# -*- coding: utf-8 -*-
"""
Deployment - Auto-deploy generated agents to Kubernetes.

This module handles production deployment of agents to Kubernetes clusters
with auto-scaling, monitoring, and rollback capabilities.

Example:
    >>> deployer = KubernetesDeployer()
    >>> config = DeploymentConfig(name="CarbonAgent", replicas=3)
    >>> deployment_id = deployer.deploy(agent_path, config)
    >>> print(f"Deployed to K8s: {deployment_id}")
"""

import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from pydantic import BaseModel, Field, validator
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class DeploymentConfig(BaseModel):
    """Configuration for agent deployment."""

    # Basic configuration
    name: str = Field(..., description="Deployment name")
    namespace: str = Field("greenlang-agents", description="Kubernetes namespace")
    replicas: int = Field(1, ge=1, le=100, description="Number of replicas")

    # Container configuration
    image: str = Field("greenlang/agent:latest", description="Docker image")
    image_pull_policy: str = Field("IfNotPresent", description="Image pull policy")
    cpu_request: str = Field("100m", description="CPU request")
    cpu_limit: str = Field("500m", description="CPU limit")
    memory_request: str = Field("128Mi", description="Memory request")
    memory_limit: str = Field("512Mi", description="Memory limit")

    # Scaling configuration
    enable_autoscaling: bool = Field(True, description="Enable horizontal pod autoscaling")
    min_replicas: int = Field(1, ge=1, description="Minimum replicas")
    max_replicas: int = Field(10, le=100, description="Maximum replicas")
    target_cpu_utilization: int = Field(70, ge=10, le=90, description="Target CPU %")

    # Network configuration
    service_type: str = Field("ClusterIP", description="Service type")
    service_port: int = Field(8080, ge=1, le=65535, description="Service port")
    container_port: int = Field(8080, ge=1, le=65535, description="Container port")

    # Health checks
    liveness_probe_enabled: bool = Field(True, description="Enable liveness probe")
    readiness_probe_enabled: bool = Field(True, description="Enable readiness probe")
    probe_initial_delay: int = Field(30, description="Probe initial delay seconds")
    probe_period: int = Field(10, description="Probe period seconds")

    # Environment
    environment: str = Field("production", description="Deployment environment")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    config_maps: List[str] = Field(default_factory=list, description="ConfigMaps to mount")
    secrets: List[str] = Field(default_factory=list, description="Secrets to mount")

    # Monitoring
    enable_monitoring: bool = Field(True, description="Enable Prometheus monitoring")
    enable_logging: bool = Field(True, description="Enable centralized logging")
    enable_tracing: bool = Field(True, description="Enable distributed tracing")

    @validator('service_type')
    def validate_service_type(cls, v):
        """Validate service type."""
        valid_types = ["ClusterIP", "NodePort", "LoadBalancer", "ExternalName"]
        if v not in valid_types:
            raise ValueError(f"Service type must be one of: {valid_types}")
        return v


class DeploymentStatus(BaseModel):
    """Deployment status information."""

    deployment_id: str = Field(..., description="Unique deployment ID")
    name: str = Field(..., description="Deployment name")
    namespace: str = Field(..., description="Kubernetes namespace")
    status: str = Field(..., description="Current status")
    replicas_ready: int = Field(0, description="Ready replicas")
    replicas_total: int = Field(0, description="Total replicas")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    endpoints: List[str] = Field(default_factory=list, description="Service endpoints")


class AgentDeployment:
    """Base deployment interface."""

    def deploy(self, agent_path: Path, config: DeploymentConfig) -> str:
        """Deploy agent to target environment."""
        raise NotImplementedError

    def status(self, deployment_id: str) -> DeploymentStatus:
        """Get deployment status."""
        raise NotImplementedError

    def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale deployment."""
        raise NotImplementedError

    def rollback(self, deployment_id: str, revision: Optional[int] = None) -> bool:
        """Rollback deployment."""
        raise NotImplementedError

    def delete(self, deployment_id: str) -> bool:
        """Delete deployment."""
        raise NotImplementedError


class KubernetesDeployer(AgentDeployment):
    """
    Deploy agents to Kubernetes clusters.

    Features:
    - Automatic manifest generation
    - Rolling updates
    - Auto-scaling configuration
    - Health check setup
    - Service exposure
    - ConfigMap/Secret management
    """

    def __init__(self, kubeconfig: Optional[Path] = None):
        """Initialize Kubernetes deployer."""
        self.kubeconfig = kubeconfig
        self.kubectl_cmd = ["kubectl"]

        if self.kubeconfig:
            self.kubectl_cmd.extend(["--kubeconfig", str(self.kubeconfig)])

        # Verify kubectl is available
        try:
            subprocess.run(
                self.kubectl_cmd + ["version", "--client"],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            logger.warning("kubectl not available - deployment will fail")

    def deploy(self, agent_path: Path, config: DeploymentConfig) -> str:
        """
        Deploy agent to Kubernetes.

        Args:
            agent_path: Path to agent code/package
            config: Deployment configuration

        Returns:
            Deployment ID
        """
        try:
            deployment_id = self._generate_deployment_id(config.name)

            # Create namespace if needed
            self._ensure_namespace(config.namespace)

            # Generate Kubernetes manifests
            manifests = self._generate_manifests(agent_path, config, deployment_id)

            # Apply manifests
            for manifest in manifests:
                self._apply_manifest(manifest, config.namespace)

            # Wait for deployment to be ready
            self._wait_for_ready(config.name, config.namespace)

            logger.info(f"Successfully deployed {config.name} as {deployment_id}")
            return deployment_id

        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}", exc_info=True)
            raise

    def status(self, deployment_id: str) -> DeploymentStatus:
        """Get deployment status from Kubernetes."""
        try:
            # Parse deployment ID to get name and namespace
            parts = deployment_id.split("-")
            name = "-".join(parts[:-1])
            namespace = "greenlang-agents"  # Default

            # Get deployment status
            result = subprocess.run(
                self.kubectl_cmd + [
                    "get", "deployment", name,
                    "-n", namespace,
                    "-o", "json"
                ],
                capture_output=True,
                text=True,
                check=True
            )

            deployment = json.loads(result.stdout)
            status_obj = deployment.get("status", {})

            # Get service endpoints
            endpoints = self._get_endpoints(name, namespace)

            return DeploymentStatus(
                deployment_id=deployment_id,
                name=name,
                namespace=namespace,
                status=self._determine_status(status_obj),
                replicas_ready=status_obj.get("readyReplicas", 0),
                replicas_total=status_obj.get("replicas", 0),
                created_at=deployment["metadata"]["creationTimestamp"],
                updated_at=DeterministicClock.utcnow(),
                endpoints=endpoints
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get status: {e.stderr}")
            raise

    def scale(self, deployment_id: str, replicas: int) -> bool:
        """Scale deployment to specified replicas."""
        try:
            parts = deployment_id.split("-")
            name = "-".join(parts[:-1])
            namespace = "greenlang-agents"

            subprocess.run(
                self.kubectl_cmd + [
                    "scale", "deployment", name,
                    f"--replicas={replicas}",
                    "-n", namespace
                ],
                check=True
            )

            logger.info(f"Scaled {deployment_id} to {replicas} replicas")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to scale: {e.stderr}")
            return False

    def rollback(self, deployment_id: str, revision: Optional[int] = None) -> bool:
        """Rollback deployment to previous or specific revision."""
        try:
            parts = deployment_id.split("-")
            name = "-".join(parts[:-1])
            namespace = "greenlang-agents"

            cmd = self.kubectl_cmd + [
                "rollout", "undo", "deployment", name,
                "-n", namespace
            ]

            if revision:
                cmd.extend(["--to-revision", str(revision)])

            subprocess.run(cmd, check=True)

            logger.info(f"Rolled back {deployment_id}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to rollback: {e.stderr}")
            return False

    def delete(self, deployment_id: str) -> bool:
        """Delete deployment and associated resources."""
        try:
            parts = deployment_id.split("-")
            name = "-".join(parts[:-1])
            namespace = "greenlang-agents"

            # Delete deployment
            subprocess.run(
                self.kubectl_cmd + [
                    "delete", "deployment", name,
                    "-n", namespace
                ],
                check=True
            )

            # Delete service
            subprocess.run(
                self.kubectl_cmd + [
                    "delete", "service", name,
                    "-n", namespace
                ],
                check=False  # Service might not exist
            )

            # Delete HPA if exists
            subprocess.run(
                self.kubectl_cmd + [
                    "delete", "hpa", name,
                    "-n", namespace
                ],
                check=False
            )

            logger.info(f"Deleted {deployment_id}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to delete: {e.stderr}")
            return False

    def _generate_deployment_id(self, name: str) -> str:
        """Generate unique deployment ID."""
        timestamp = DeterministicClock.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{name}-{timestamp}"

    def _ensure_namespace(self, namespace: str):
        """Create namespace if it doesn't exist."""
        subprocess.run(
            self.kubectl_cmd + [
                "create", "namespace", namespace,
                "--dry-run=client", "-o", "yaml"
            ],
            capture_output=True
        )

    def _generate_manifests(
        self,
        agent_path: Path,
        config: DeploymentConfig,
        deployment_id: str
    ) -> List[Dict[str, Any]]:
        """Generate Kubernetes manifests."""
        manifests = []

        # Deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace,
                "labels": {
                    "app": config.name,
                    "deployment-id": deployment_id,
                    "managed-by": "greenlang-factory"
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.name,
                            "version": "1.0.0"
                        },
                        "annotations": {}
                    },
                    "spec": {
                        "containers": [{
                            "name": config.name,
                            "image": config.image,
                            "imagePullPolicy": config.image_pull_policy,
                            "ports": [{
                                "containerPort": config.container_port,
                                "protocol": "TCP"
                            }],
                            "resources": {
                                "requests": {
                                    "cpu": config.cpu_request,
                                    "memory": config.memory_request
                                },
                                "limits": {
                                    "cpu": config.cpu_limit,
                                    "memory": config.memory_limit
                                }
                            },
                            "env": [
                                {"name": k, "value": v}
                                for k, v in config.env_vars.items()
                            ]
                        }]
                    }
                }
            }
        }

        # Add probes if enabled
        container = deployment["spec"]["template"]["spec"]["containers"][0]

        if config.liveness_probe_enabled:
            container["livenessProbe"] = {
                "httpGet": {
                    "path": "/health",
                    "port": config.container_port
                },
                "initialDelaySeconds": config.probe_initial_delay,
                "periodSeconds": config.probe_period
            }

        if config.readiness_probe_enabled:
            container["readinessProbe"] = {
                "httpGet": {
                    "path": "/ready",
                    "port": config.container_port
                },
                "initialDelaySeconds": config.probe_initial_delay,
                "periodSeconds": config.probe_period
            }

        # Add monitoring annotations
        if config.enable_monitoring:
            deployment["spec"]["template"]["metadata"]["annotations"].update({
                "prometheus.io/scrape": "true",
                "prometheus.io/port": str(config.container_port),
                "prometheus.io/path": "/metrics"
            })

        manifests.append(deployment)

        # Service manifest
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace,
                "labels": {
                    "app": config.name
                }
            },
            "spec": {
                "type": config.service_type,
                "selector": {
                    "app": config.name
                },
                "ports": [{
                    "port": config.service_port,
                    "targetPort": config.container_port,
                    "protocol": "TCP"
                }]
            }
        }

        manifests.append(service)

        # HPA manifest if autoscaling enabled
        if config.enable_autoscaling:
            hpa = {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": config.name,
                    "namespace": config.namespace
                },
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": config.name
                    },
                    "minReplicas": config.min_replicas,
                    "maxReplicas": config.max_replicas,
                    "metrics": [{
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": config.target_cpu_utilization
                            }
                        }
                    }]
                }
            }

            manifests.append(hpa)

        return manifests

    def _apply_manifest(self, manifest: Dict[str, Any], namespace: str):
        """Apply manifest to Kubernetes."""
        # Convert to YAML
        yaml_content = yaml.dump(manifest)

        # Apply using kubectl
        process = subprocess.Popen(
            self.kubectl_cmd + ["apply", "-f", "-", "-n", namespace],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=yaml_content)

        if process.returncode != 0:
            raise RuntimeError(f"Failed to apply manifest: {stderr}")

    def _wait_for_ready(self, name: str, namespace: str, timeout: int = 300):
        """Wait for deployment to be ready."""
        subprocess.run(
            self.kubectl_cmd + [
                "wait", "--for=condition=available",
                f"deployment/{name}",
                "-n", namespace,
                f"--timeout={timeout}s"
            ],
            check=True
        )

    def _get_endpoints(self, name: str, namespace: str) -> List[str]:
        """Get service endpoints."""
        try:
            result = subprocess.run(
                self.kubectl_cmd + [
                    "get", "endpoints", name,
                    "-n", namespace,
                    "-o", "json"
                ],
                capture_output=True,
                text=True,
                check=True
            )

            endpoints_obj = json.loads(result.stdout)
            endpoints = []

            for subset in endpoints_obj.get("subsets", []):
                for address in subset.get("addresses", []):
                    for port in subset.get("ports", []):
                        endpoints.append(f"{address['ip']}:{port['port']}")

            return endpoints

        except Exception:
            return []

    def _determine_status(self, status_obj: Dict[str, Any]) -> str:
        """Determine deployment status from K8s status object."""
        conditions = status_obj.get("conditions", [])

        for condition in conditions:
            if condition["type"] == "Progressing" and condition["status"] == "True":
                if "successfully progressed" in condition.get("reason", "").lower():
                    return "Running"
                return "Updating"

        ready_replicas = status_obj.get("readyReplicas", 0)
        replicas = status_obj.get("replicas", 1)

        if ready_replicas == replicas:
            return "Running"
        elif ready_replicas > 0:
            return "Partial"
        else:
            return "Pending"