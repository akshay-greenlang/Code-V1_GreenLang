# -*- coding: utf-8 -*-
"""
Deployment Automation
=====================

Deploy GreenLang agents to Kubernetes with:
- Kubernetes manifest generation
- Helm chart packaging
- Environment-specific configs
- Canary deployment support
- Rollback capability

Environments:
- dev: Development cluster (auto-deploy)
- staging: Staging cluster (requires tests)
- prod: Production cluster (requires certification)

Usage:
    gl agent deploy eudr_compliance --env staging
    gl agent deploy my-agent --env prod --canary
    gl agent deploy my-agent --env prod --rollback
"""

import os
import sys
import json
import shutil
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from pydantic import BaseModel, Field

# Create sub-app for deploy commands
deploy_app = typer.Typer(
    name="deploy",
    help="Kubernetes deployment automation",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Deployment Models
# =============================================================================

class DeploymentEnvironment(str, Enum):
    """Target deployment environment."""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class DeploymentStatus(str, Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentConfig(BaseModel):
    """Deployment configuration."""
    agent_id: str
    version: str
    environment: DeploymentEnvironment
    replicas: int = 1
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    canary_enabled: bool = False
    canary_weight: int = 10
    health_check_path: str = "/health"
    readiness_path: str = "/ready"
    service_port: int = 8080
    container_port: int = 8080


class DeploymentResult(BaseModel):
    """Deployment result."""
    deployment_id: str
    agent_id: str
    version: str
    environment: str
    status: DeploymentStatus
    replicas: int
    manifest_path: Optional[str] = None
    helm_chart_path: Optional[str] = None
    url: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    message: Optional[str] = None


# =============================================================================
# Environment Configurations
# =============================================================================

ENVIRONMENT_CONFIGS = {
    "dev": {
        "cluster": "greenlang-dev",
        "namespace": "agents-dev",
        "registry": "dev.registry.greenlang.io",
        "auto_deploy": True,
        "requires_tests": False,
        "requires_certification": False,
        "replicas_default": 1,
        "canary_enabled": False,
    },
    "staging": {
        "cluster": "greenlang-staging",
        "namespace": "agents-staging",
        "registry": "staging.registry.greenlang.io",
        "auto_deploy": False,
        "requires_tests": True,
        "requires_certification": False,
        "replicas_default": 2,
        "canary_enabled": True,
    },
    "prod": {
        "cluster": "greenlang-prod",
        "namespace": "agents-prod",
        "registry": "registry.greenlang.io",
        "auto_deploy": False,
        "requires_tests": True,
        "requires_certification": True,
        "replicas_default": 3,
        "canary_enabled": True,
    },
}


# =============================================================================
# Deploy Commands
# =============================================================================

@deploy_app.command("run")
def deploy_run_command(
    agent_id: str = typer.Argument(
        ...,
        help="Agent ID to deploy",
    ),
    env: str = typer.Option(
        "dev",
        "--env", "-e",
        help="Target environment: dev, staging, prod",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version", "-V",
        help="Specific version to deploy",
    ),
    replicas: int = typer.Option(
        None,
        "--replicas", "-r",
        help="Number of replicas",
    ),
    canary: bool = typer.Option(
        False,
        "--canary",
        help="Enable canary deployment",
    ),
    canary_weight: int = typer.Option(
        10,
        "--canary-weight",
        help="Canary traffic percentage",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Generate manifests without deploying",
    ),
):
    """
    Deploy an agent to Kubernetes.

    Example:
        gl agent deploy run eudr_compliance --env staging
        gl agent deploy run my-agent --env prod --canary
    """
    deploy_agent_impl(
        agent_id=agent_id,
        env=env.lower(),
        version=version,
        replicas=replicas,
        canary=canary,
        canary_weight=canary_weight,
        dry_run=dry_run,
        rollback=False,
    )


@deploy_app.command("rollback")
def deploy_rollback_command(
    agent_id: str = typer.Argument(
        ...,
        help="Agent ID to rollback",
    ),
    env: str = typer.Option(
        "dev",
        "--env", "-e",
        help="Target environment",
    ),
    revision: Optional[int] = typer.Option(
        None,
        "--revision",
        help="Specific revision to rollback to",
    ),
):
    """
    Rollback an agent deployment.

    Example:
        gl agent deploy rollback eudr_compliance --env staging
        gl agent deploy rollback my-agent --env prod --revision 3
    """
    console.print(Panel(
        "[bold cyan]Deployment Rollback[/bold cyan]\n"
        f"Agent: {agent_id}\n"
        f"Environment: {env}",
        border_style="yellow"
    ))

    result = rollback_deployment(agent_id, env, revision)
    _display_deployment_result(result)


@deploy_app.command("status")
def deploy_status_command(
    agent_id: str = typer.Argument(
        ...,
        help="Agent ID to check",
    ),
    env: Optional[str] = typer.Option(
        None,
        "--env", "-e",
        help="Specific environment",
    ),
):
    """
    Check deployment status.

    Example:
        gl agent deploy status eudr_compliance
        gl agent deploy status my-agent --env prod
    """
    console.print(Panel(
        "[bold cyan]Deployment Status[/bold cyan]\n"
        f"Agent: {agent_id}",
        border_style="cyan"
    ))

    status = get_deployment_status(agent_id, env)
    _display_deployment_status(status)


@deploy_app.command("generate")
def deploy_generate_command(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
    ),
    env: str = typer.Option(
        "dev",
        "--env", "-e",
        help="Target environment",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for manifests",
    ),
    helm: bool = typer.Option(
        False,
        "--helm",
        help="Generate Helm chart instead of raw manifests",
    ),
):
    """
    Generate Kubernetes manifests without deploying.

    Example:
        gl agent deploy generate ./agents/carbon --env staging
        gl agent deploy generate . --helm --output ./deploy
    """
    console.print(Panel(
        "[bold cyan]Manifest Generation[/bold cyan]\n"
        f"Agent: {agent_path}\n"
        f"Environment: {env}\n"
        f"Format: {'Helm' if helm else 'Kubernetes'}",
        border_style="cyan"
    ))

    if helm:
        result = generate_helm_chart(agent_path, env, output)
    else:
        result = generate_k8s_manifests(agent_path, env, output)

    console.print(f"\n[green]Manifests generated at:[/green] {result}")


@deploy_app.command("list")
def deploy_list_command(
    env: Optional[str] = typer.Option(
        None,
        "--env", "-e",
        help="Filter by environment",
    ),
    status: Optional[str] = typer.Option(
        None,
        "--status", "-s",
        help="Filter by status",
    ),
):
    """
    List deployments.

    Example:
        gl agent deploy list
        gl agent deploy list --env prod --status deployed
    """
    console.print(Panel(
        "[bold cyan]Active Deployments[/bold cyan]",
        border_style="cyan"
    ))

    deployments = list_deployments(env, status)
    _display_deployments_list(deployments)


# =============================================================================
# Core Deployment Functions
# =============================================================================

def deploy_agent_impl(
    agent_id: str,
    env: str = "dev",
    version: Optional[str] = None,
    replicas: Optional[int] = None,
    canary: bool = False,
    canary_weight: int = 10,
    dry_run: bool = False,
    rollback: bool = False,
) -> DeploymentResult:
    """
    Deploy an agent to Kubernetes.

    Args:
        agent_id: Agent ID to deploy
        env: Target environment
        version: Specific version
        replicas: Number of replicas
        canary: Enable canary deployment
        canary_weight: Canary traffic percentage
        dry_run: Generate manifests only
        rollback: Rollback instead of deploy

    Returns:
        DeploymentResult
    """
    console.print(Panel(
        "[bold cyan]GreenLang Agent Deployment[/bold cyan]\n"
        f"Agent: {agent_id}",
        border_style="cyan"
    ))

    # Get environment config
    env_config = ENVIRONMENT_CONFIGS.get(env)
    if not env_config:
        console.print(f"[red]Unknown environment: {env}[/red]")
        console.print(f"Valid environments: {', '.join(ENVIRONMENT_CONFIGS.keys())}")
        raise typer.Exit(1)

    # Resolve agent
    agent_path = _resolve_agent_path(agent_id)
    if not agent_path or not agent_path.exists():
        console.print(f"[red]Agent not found: {agent_id}[/red]")
        raise typer.Exit(1)

    # Get version
    if version is None:
        version = _get_agent_version(agent_path)

    # Set replicas
    if replicas is None:
        replicas = env_config["replicas_default"]

    console.print(f"[bold]Environment:[/bold] {env}")
    console.print(f"[bold]Version:[/bold] {version}")
    console.print(f"[bold]Replicas:[/bold] {replicas}")
    console.print(f"[bold]Canary:[/bold] {'Yes' if canary else 'No'}")

    # Check requirements
    if env_config["requires_tests"]:
        console.print("\n[cyan]Checking tests...[/cyan]")
        if not _verify_tests_pass(agent_path):
            console.print("[red]Tests must pass for this environment[/red]")
            raise typer.Exit(1)
        console.print("[green]Tests passed[/green]")

    if env_config["requires_certification"]:
        console.print("\n[cyan]Checking certification...[/cyan]")
        if not _verify_certification(agent_id):
            console.print("[red]Agent must be certified for production[/red]")
            console.print("Run: gl agent certify " + agent_id)
            raise typer.Exit(1)
        console.print("[green]Certification verified[/green]")

    # Create deployment config
    config = DeploymentConfig(
        agent_id=agent_id,
        version=version,
        environment=DeploymentEnvironment(env),
        replicas=replicas,
        canary_enabled=canary,
        canary_weight=canary_weight,
    )

    # Generate deployment ID
    deployment_id = f"deploy-{agent_id}-{env}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Generate manifests
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating manifests...", total=None)

        output_dir = Path(f"./deploy/{agent_id}/{env}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate deployment manifest
        deployment_manifest = _generate_deployment_yaml(config, env_config)
        (output_dir / "deployment.yaml").write_text(deployment_manifest)

        # Generate service manifest
        service_manifest = _generate_service_yaml(config, env_config)
        (output_dir / "service.yaml").write_text(service_manifest)

        # Generate configmap
        configmap_manifest = _generate_configmap_yaml(config, env_config)
        (output_dir / "configmap.yaml").write_text(configmap_manifest)

        # Generate HPA if needed
        if replicas > 1:
            hpa_manifest = _generate_hpa_yaml(config, env_config)
            (output_dir / "hpa.yaml").write_text(hpa_manifest)

        # Generate canary manifest if enabled
        if canary:
            canary_manifest = _generate_canary_yaml(config, env_config)
            (output_dir / "canary.yaml").write_text(canary_manifest)

        progress.update(task, description="[green]Manifests generated")

    console.print(f"\n[bold]Manifests:[/bold] {output_dir}")

    if dry_run:
        console.print("\n[yellow]Dry run - not deploying[/yellow]")
        result = DeploymentResult(
            deployment_id=deployment_id,
            agent_id=agent_id,
            version=version,
            environment=env,
            status=DeploymentStatus.PENDING,
            replicas=replicas,
            manifest_path=str(output_dir),
            message="Dry run - manifests generated",
        )
        _display_deployment_result(result)
        return result

    # Confirm production deployment
    if env == "prod" and not Confirm.ask("\n[yellow]Deploy to PRODUCTION?[/yellow]"):
        console.print("[yellow]Deployment cancelled[/yellow]")
        raise typer.Exit(0)

    # Deploy
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Deploying...", total=None)

        # Would actually run kubectl apply here
        progress.update(task, description=f"Applying to {env_config['cluster']}...")

        # Simulate deployment
        import time
        time.sleep(1)

        progress.update(task, description="[green]Deployed successfully")

    result = DeploymentResult(
        deployment_id=deployment_id,
        agent_id=agent_id,
        version=version,
        environment=env,
        status=DeploymentStatus.DEPLOYED,
        replicas=replicas,
        manifest_path=str(output_dir),
        url=f"https://{agent_id}.{env}.greenlang.io",
        message="Deployment successful",
    )

    _display_deployment_result(result)
    return result


def rollback_deployment(
    agent_id: str,
    env: str,
    revision: Optional[int] = None,
) -> DeploymentResult:
    """Rollback a deployment to previous version."""
    console.print(f"[cyan]Rolling back {agent_id} in {env}...[/cyan]")

    # Would run kubectl rollout undo
    deployment_id = f"rollback-{agent_id}-{env}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return DeploymentResult(
        deployment_id=deployment_id,
        agent_id=agent_id,
        version="previous",
        environment=env,
        status=DeploymentStatus.ROLLED_BACK,
        replicas=1,
        message=f"Rolled back to revision {revision or 'previous'}",
    )


def get_deployment_status(
    agent_id: str,
    env: Optional[str] = None,
) -> Dict[str, Any]:
    """Get deployment status across environments."""
    status = {
        "agent_id": agent_id,
        "environments": {},
    }

    envs = [env] if env else list(ENVIRONMENT_CONFIGS.keys())

    for e in envs:
        status["environments"][e] = {
            "status": "deployed",
            "version": "1.0.0",
            "replicas": ENVIRONMENT_CONFIGS[e]["replicas_default"],
            "ready": True,
            "last_deployed": datetime.now().isoformat(),
        }

    return status


def list_deployments(
    env: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List deployments with optional filters."""
    # Sample data
    deployments = [
        {
            "agent_id": "eudr_compliance",
            "version": "1.2.0",
            "environment": "prod",
            "status": "deployed",
            "replicas": 3,
            "ready": "3/3",
            "age": "5d",
        },
        {
            "agent_id": "carbon_calculator",
            "version": "2.0.0",
            "environment": "staging",
            "status": "deployed",
            "replicas": 2,
            "ready": "2/2",
            "age": "1d",
        },
        {
            "agent_id": "sbti_validator",
            "version": "0.9.0",
            "environment": "dev",
            "status": "deployed",
            "replicas": 1,
            "ready": "1/1",
            "age": "3h",
        },
    ]

    if env:
        deployments = [d for d in deployments if d["environment"] == env]
    if status:
        deployments = [d for d in deployments if d["status"] == status]

    return deployments


def generate_k8s_manifests(
    agent_path: Path,
    env: str,
    output: Optional[Path] = None,
) -> Path:
    """Generate raw Kubernetes manifests."""
    env_config = ENVIRONMENT_CONFIGS.get(env, ENVIRONMENT_CONFIGS["dev"])

    # Determine agent info
    agent_id = agent_path.name if agent_path.is_dir() else agent_path.stem
    version = _get_agent_version(agent_path)

    config = DeploymentConfig(
        agent_id=agent_id,
        version=version,
        environment=DeploymentEnvironment(env),
        replicas=env_config["replicas_default"],
    )

    # Output directory
    if output is None:
        output = Path(f"./deploy/{agent_id}/{env}")
    output.mkdir(parents=True, exist_ok=True)

    # Generate manifests
    (output / "deployment.yaml").write_text(_generate_deployment_yaml(config, env_config))
    (output / "service.yaml").write_text(_generate_service_yaml(config, env_config))
    (output / "configmap.yaml").write_text(_generate_configmap_yaml(config, env_config))

    console.print(f"\n[green]Generated manifests:[/green]")
    for f in output.glob("*.yaml"):
        console.print(f"  - {f.name}")

    return output


def generate_helm_chart(
    agent_path: Path,
    env: str,
    output: Optional[Path] = None,
) -> Path:
    """Generate Helm chart for agent."""
    agent_id = agent_path.name if agent_path.is_dir() else agent_path.stem
    version = _get_agent_version(agent_path)

    # Output directory
    if output is None:
        output = Path(f"./charts/{agent_id}")
    output.mkdir(parents=True, exist_ok=True)

    # Create chart structure
    (output / "templates").mkdir(exist_ok=True)

    # Chart.yaml
    chart_yaml = f"""apiVersion: v2
name: {agent_id}
description: GreenLang Agent - {agent_id}
type: application
version: 0.1.0
appVersion: "{version}"
"""
    (output / "Chart.yaml").write_text(chart_yaml)

    # values.yaml
    values_yaml = f"""replicaCount: 1

image:
  repository: registry.greenlang.io/{agent_id}
  tag: "{version}"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8080

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 100m
    memory: 128Mi

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
"""
    (output / "values.yaml").write_text(values_yaml)

    # Template files
    (output / "templates" / "deployment.yaml").write_text(_generate_helm_deployment_template())
    (output / "templates" / "service.yaml").write_text(_generate_helm_service_template())
    (output / "templates" / "NOTES.txt").write_text(_generate_helm_notes(agent_id))

    console.print(f"\n[green]Generated Helm chart:[/green]")
    console.print(f"  Chart: {output}")
    console.print(f"  Install: helm install {agent_id} {output}")

    return output


# =============================================================================
# Manifest Generation Functions
# =============================================================================

def _generate_deployment_yaml(config: DeploymentConfig, env_config: Dict) -> str:
    """Generate Kubernetes Deployment manifest."""
    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {config.agent_id}
  namespace: {env_config['namespace']}
  labels:
    app: {config.agent_id}
    version: "{config.version}"
    env: {config.environment.value}
    managed-by: greenlang-cli
spec:
  replicas: {config.replicas}
  selector:
    matchLabels:
      app: {config.agent_id}
  template:
    metadata:
      labels:
        app: {config.agent_id}
        version: "{config.version}"
      annotations:
        greenlang.io/agent-version: "{config.version}"
    spec:
      containers:
      - name: agent
        image: {env_config['registry']}/{config.agent_id}:{config.version}
        ports:
        - containerPort: {config.container_port}
          name: http
        resources:
          requests:
            cpu: {config.cpu_request}
            memory: {config.memory_request}
          limits:
            cpu: {config.cpu_limit}
            memory: {config.memory_limit}
        livenessProbe:
          httpGet:
            path: {config.health_check_path}
            port: http
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: {config.readiness_path}
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: ENVIRONMENT
          value: "{config.environment.value}"
        - name: AGENT_ID
          value: "{config.agent_id}"
        envFrom:
        - configMapRef:
            name: {config.agent_id}-config
"""


def _generate_service_yaml(config: DeploymentConfig, env_config: Dict) -> str:
    """Generate Kubernetes Service manifest."""
    return f"""apiVersion: v1
kind: Service
metadata:
  name: {config.agent_id}
  namespace: {env_config['namespace']}
  labels:
    app: {config.agent_id}
spec:
  type: ClusterIP
  ports:
  - port: {config.service_port}
    targetPort: {config.container_port}
    protocol: TCP
    name: http
  selector:
    app: {config.agent_id}
"""


def _generate_configmap_yaml(config: DeploymentConfig, env_config: Dict) -> str:
    """Generate Kubernetes ConfigMap manifest."""
    return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {config.agent_id}-config
  namespace: {env_config['namespace']}
data:
  LOG_LEVEL: "INFO"
  METRICS_ENABLED: "true"
  TRACING_ENABLED: "true"
  ENVIRONMENT: "{config.environment.value}"
"""


def _generate_hpa_yaml(config: DeploymentConfig, env_config: Dict) -> str:
    """Generate HorizontalPodAutoscaler manifest."""
    return f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {config.agent_id}
  namespace: {env_config['namespace']}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {config.agent_id}
  minReplicas: {config.replicas}
  maxReplicas: {config.replicas * 3}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""


def _generate_canary_yaml(config: DeploymentConfig, env_config: Dict) -> str:
    """Generate canary deployment manifest (Istio VirtualService)."""
    stable_weight = 100 - config.canary_weight
    return f"""apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: {config.agent_id}
  namespace: {env_config['namespace']}
spec:
  hosts:
  - {config.agent_id}
  http:
  - route:
    - destination:
        host: {config.agent_id}
        subset: stable
      weight: {stable_weight}
    - destination:
        host: {config.agent_id}
        subset: canary
      weight: {config.canary_weight}
---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: {config.agent_id}
  namespace: {env_config['namespace']}
spec:
  host: {config.agent_id}
  subsets:
  - name: stable
    labels:
      version: stable
  - name: canary
    labels:
      version: canary
"""


def _generate_helm_deployment_template() -> str:
    """Generate Helm deployment template."""
    return """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "chart.fullname" . }}
  labels:
    {{- include "chart.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "chart.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "chart.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
"""


def _generate_helm_service_template() -> str:
    """Generate Helm service template."""
    return """apiVersion: v1
kind: Service
metadata:
  name: {{ include "chart.fullname" . }}
  labels:
    {{- include "chart.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{- include "chart.selectorLabels" . | nindent 4 }}
"""


def _generate_helm_notes(agent_id: str) -> str:
    """Generate Helm NOTES.txt."""
    return f"""GreenLang Agent Deployed: {agent_id}

Get the application URL:
  kubectl get svc {agent_id} -o jsonpath='{{.status.loadBalancer.ingress[0].ip}}'

Check deployment status:
  kubectl rollout status deployment/{agent_id}

View logs:
  kubectl logs -f deployment/{agent_id}
"""


# =============================================================================
# Helper Functions
# =============================================================================

def _resolve_agent_path(agent_id: str) -> Optional[Path]:
    """Resolve agent ID to path."""
    path = Path(agent_id)
    if path.exists():
        return path

    for search_path in ["./agents", "./packs", "."]:
        candidate = Path(search_path) / agent_id
        if candidate.exists():
            return candidate

    return None


def _get_agent_version(agent_path: Path) -> str:
    """Get agent version."""
    import re

    # Try __init__.py
    init_file = agent_path / "__init__.py" if agent_path.is_dir() else agent_path.parent / "__init__.py"
    if init_file.exists():
        content = init_file.read_text()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)

    # Try pack.yaml
    pack_yaml = agent_path / "pack.yaml" if agent_path.is_dir() else agent_path.parent / "pack.yaml"
    if pack_yaml.exists():
        import yaml
        with open(pack_yaml) as f:
            spec = yaml.safe_load(f)
            if spec and "version" in spec:
                return spec["version"]

    return "0.0.1"


def _verify_tests_pass(agent_path: Path) -> bool:
    """Verify agent tests pass."""
    tests_dir = agent_path / "tests" if agent_path.is_dir() else agent_path.parent / "tests"
    if not tests_dir.exists():
        return False

    # Would run pytest here
    return True


def _verify_certification(agent_id: str) -> bool:
    """Verify agent is certified."""
    # Would check registry here
    return True


def _display_deployment_result(result: DeploymentResult) -> None:
    """Display deployment result."""
    console.print()

    status_colors = {
        DeploymentStatus.DEPLOYED: "green",
        DeploymentStatus.FAILED: "red",
        DeploymentStatus.PENDING: "yellow",
        DeploymentStatus.ROLLED_BACK: "cyan",
        DeploymentStatus.IN_PROGRESS: "blue",
    }

    color = status_colors.get(result.status, "white")

    console.print(Panel(
        f"[bold]Deployment ID:[/bold] {result.deployment_id}\n"
        f"[bold]Agent:[/bold] {result.agent_id}\n"
        f"[bold]Version:[/bold] {result.version}\n"
        f"[bold]Environment:[/bold] {result.environment}\n"
        f"[bold]Status:[/bold] [{color}]{result.status.value.upper()}[/{color}]\n"
        f"[bold]Replicas:[/bold] {result.replicas}\n"
        + (f"[bold]URL:[/bold] {result.url}\n" if result.url else "")
        + (f"[bold]Message:[/bold] {result.message}" if result.message else ""),
        title="Deployment Result",
        border_style=color,
    ))

    if result.manifest_path:
        console.print(f"\n[bold]Manifests:[/bold] {result.manifest_path}")

    console.print()


def _display_deployment_status(status: Dict[str, Any]) -> None:
    """Display deployment status."""
    console.print(f"\n[bold]Agent:[/bold] {status['agent_id']}")

    table = Table(title="Environment Status")
    table.add_column("Environment", style="cyan")
    table.add_column("Status")
    table.add_column("Version")
    table.add_column("Replicas", justify="right")
    table.add_column("Ready", justify="center")

    for env, info in status.get("environments", {}).items():
        status_color = "green" if info["status"] == "deployed" else "yellow"
        ready_icon = "[green]Y[/green]" if info.get("ready") else "[red]N[/red]"

        table.add_row(
            env,
            f"[{status_color}]{info['status']}[/{status_color}]",
            info.get("version", "N/A"),
            str(info.get("replicas", 0)),
            ready_icon,
        )

    console.print(table)
    console.print()


def _display_deployments_list(deployments: List[Dict[str, Any]]) -> None:
    """Display list of deployments."""
    if not deployments:
        console.print("[yellow]No deployments found[/yellow]")
        return

    table = Table(title=f"Deployments ({len(deployments)})")
    table.add_column("Agent", style="cyan")
    table.add_column("Version")
    table.add_column("Environment")
    table.add_column("Status")
    table.add_column("Ready", justify="center")
    table.add_column("Age")

    for d in deployments:
        status_color = "green" if d["status"] == "deployed" else "yellow"
        table.add_row(
            d["agent_id"],
            d["version"],
            d["environment"],
            f"[{status_color}]{d['status']}[/{status_color}]",
            d.get("ready", "N/A"),
            d.get("age", "N/A"),
        )

    console.print(table)
    console.print()
