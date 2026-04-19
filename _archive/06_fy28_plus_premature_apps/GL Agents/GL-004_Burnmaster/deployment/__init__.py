"""GL-004 BURNMASTER Deployment Module"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"

class DeploymentTarget(Enum):
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker-compose"
    HELM = "helm"

@dataclass
class DeploymentConfig:
    environment: Environment
    target: DeploymentTarget
    namespace: str = "greenlang"
    replicas: int = 3
    image_tag: str = "latest"
    registry: str = "gcr.io/greenlang"
    resource_limits: Dict[str, str] = field(default_factory=lambda: {"cpu": "2000m", "memory": "4Gi"})
    resource_requests: Dict[str, str] = field(default_factory=lambda: {"cpu": "500m", "memory": "1Gi"})

@dataclass
class ServiceEndpoint:
    host: str
    port: int
    protocol: str = "http"
    tls_enabled: bool = False

class ConfigLoader:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent / "config"
        self._cache: Dict[str, Any] = {}

class SecretManager:
    @staticmethod
    def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(name, default)

def get_deployment_config(environment: Optional[str] = None, target: Optional[str] = None) -> DeploymentConfig:
    env = Environment(environment or os.getenv("GL_ENVIRONMENT", "development"))
    tgt = DeploymentTarget(target or os.getenv("GL_DEPLOY_TARGET", "kubernetes"))
    return DeploymentConfig(environment=env, target=tgt)

__all__ = ["Environment", "DeploymentTarget", "DeploymentConfig", "ServiceEndpoint", "ConfigLoader", "SecretManager", "get_deployment_config"]
__version__ = "1.0.0"
