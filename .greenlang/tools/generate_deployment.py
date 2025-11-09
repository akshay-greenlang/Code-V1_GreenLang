#!/usr/bin/env python3
"""
Deployment Configuration Generator

Generate deployment configurations for Docker, Kubernetes, and Terraform.
Supports multi-environment deployments with best practices.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any


class DeploymentGenerator:
    """Generate deployment configurations."""

    @staticmethod
    def generate_kubernetes_manifests(app_name: str, features: Dict[str, bool] = None) -> Dict[str, str]:
        """Generate Kubernetes manifests."""

        if features is None:
            features = {}

        manifests = {}

        # Deployment
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': app_name,
                'labels': {
                    'app': app_name
                }
            },
            'spec': {
                'replicas': 3,
                'selector': {
                    'matchLabels': {
                        'app': app_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': app_name
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': app_name,
                            'image': f'{app_name}:latest',
                            'ports': [{'containerPort': 8000}],
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': 'production'},
                                {'name': 'APP_NAME', 'value': app_name}
                            ],
                            'envFrom': [
                                {'secretRef': {'name': f'{app_name}-secrets'}}
                            ],
                            'resources': {
                                'requests': {
                                    'memory': '256Mi',
                                    'cpu': '250m'
                                },
                                'limits': {
                                    'memory': '512Mi',
                                    'cpu': '500m'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 10,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }

        manifests['deployment.yaml'] = yaml.dump(deployment, default_flow_style=False, sort_keys=False)

        # Service
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': app_name,
                'labels': {
                    'app': app_name
                }
            },
            'spec': {
                'type': 'LoadBalancer',
                'ports': [{
                    'port': 80,
                    'targetPort': 8000,
                    'protocol': 'TCP'
                }],
                'selector': {
                    'app': app_name
                }
            }
        }

        manifests['service.yaml'] = yaml.dump(service, default_flow_style=False, sort_keys=False)

        # ConfigMap
        configmap = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f'{app_name}-config'
            },
            'data': {
                'LOG_LEVEL': 'INFO',
                'CACHE_TYPE': 'redis',
                'CACHE_TTL': '3600'
            }
        }

        manifests['configmap.yaml'] = yaml.dump(configmap, default_flow_style=False, sort_keys=False)

        # Secret (template)
        secret = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': f'{app_name}-secrets'
            },
            'type': 'Opaque',
            'stringData': {
                'OPENAI_API_KEY': 'your-key-here',
                'DATABASE_URL': 'postgresql://user:pass@host:5432/db'
            }
        }

        manifests['secret.yaml.template'] = yaml.dump(secret, default_flow_style=False, sort_keys=False)

        # HPA (Horizontal Pod Autoscaler)
        hpa = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': app_name
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': app_name
                },
                'minReplicas': 2,
                'maxReplicas': 10,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    }
                ]
            }
        }

        manifests['hpa.yaml'] = yaml.dump(hpa, default_flow_style=False, sort_keys=False)

        return manifests

    @staticmethod
    def generate_terraform(app_name: str) -> Dict[str, str]:
        """Generate Terraform configurations."""

        files = {}

        # Main configuration
        files['main.tf'] = f'''terraform {{
  required_version = ">= 1.0"

  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
    kubernetes = {{
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }}
  }}

  backend "s3" {{
    bucket = "{app_name}-terraform-state"
    key    = "terraform.tfstate"
    region = "us-east-1"
  }}
}}

provider "aws" {{
  region = var.aws_region
}}

provider "kubernetes" {{
  config_path = "~/.kube/config"
}}
'''

        # Variables
        files['variables.tf'] = f'''variable "aws_region" {{
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}}

variable "app_name" {{
  description = "Application name"
  type        = string
  default     = "{app_name}"
}}

variable "environment" {{
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "production"
}}

variable "replicas" {{
  description = "Number of replicas"
  type        = number
  default     = 3
}}

variable "instance_type" {{
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}}
'''

        # Resources
        files['resources.tf'] = f'''# ECS Cluster
resource "aws_ecs_cluster" "main" {{
  name = "${{var.app_name}}-${{var.environment}}"

  setting {{
    name  = "containerInsights"
    value = "enabled"
  }}

  tags = {{
    Name        = var.app_name
    Environment = var.environment
  }}
}}

# ECR Repository
resource "aws_ecr_repository" "app" {{
  name                 = var.app_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {{
    scan_on_push = true
  }}

  tags = {{
    Name        = var.app_name
    Environment = var.environment
  }}
}}

# VPC
resource "aws_vpc" "main" {{
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {{
    Name        = "${{var.app_name}}-vpc"
    Environment = var.environment
  }}
}}

# Subnets
resource "aws_subnet" "public" {{
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${{count.index}}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {{
    Name        = "${{var.app_name}}-public-${{count.index}}"
    Environment = var.environment
  }}
}}

data "aws_availability_zones" "available" {{
  state = "available"
}}
'''

        # Outputs
        files['outputs.tf'] = '''output "cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.app.repository_url
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}
'''

        return files

    @staticmethod
    def generate_docker_compose_production(app_name: str, features: Dict[str, bool] = None) -> str:
        """Generate production docker-compose.yml."""

        if features is None:
            features = {}

        compose = {
            'version': '3.8',
            'services': {
                'app': {
                    'image': f'{app_name}:latest',
                    'container_name': app_name,
                    'restart': 'always',
                    'environment': [
                        'ENVIRONMENT=production'
                    ],
                    'env_file': ['.env.production'],
                    'ports': ['8000:8000'],
                    'networks': ['app-network'],
                    'depends_on': ['redis', 'postgres'],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '40s'
                    },
                    'deploy': {
                        'replicas': 3,
                        'restart_policy': {
                            'condition': 'on-failure',
                            'delay': '5s',
                            'max_attempts': 3
                        },
                        'resources': {
                            'limits': {
                                'cpus': '1.0',
                                'memory': '512M'
                            },
                            'reservations': {
                                'cpus': '0.5',
                                'memory': '256M'
                            }
                        }
                    }
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'container_name': f'{app_name}-redis',
                    'restart': 'always',
                    'ports': ['6379:6379'],
                    'networks': ['app-network'],
                    'volumes': ['redis_data:/data'],
                    'command': 'redis-server --appendonly yes'
                },
                'postgres': {
                    'image': 'postgres:15-alpine',
                    'container_name': f'{app_name}-postgres',
                    'restart': 'always',
                    'environment': [
                        'POSTGRES_USER=${DB_USER}',
                        'POSTGRES_PASSWORD=${DB_PASSWORD}',
                        'POSTGRES_DB=${DB_NAME}'
                    ],
                    'ports': ['5432:5432'],
                    'networks': ['app-network'],
                    'volumes': ['postgres_data:/var/lib/postgresql/data']
                },
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'container_name': f'{app_name}-prometheus',
                    'restart': 'always',
                    'ports': ['9090:9090'],
                    'networks': ['app-network'],
                    'volumes': [
                        './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml',
                        'prometheus_data:/prometheus'
                    ]
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'container_name': f'{app_name}-grafana',
                    'restart': 'always',
                    'ports': ['3000:3000'],
                    'networks': ['app-network'],
                    'environment': [
                        'GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}'
                    ],
                    'volumes': ['grafana_data:/var/lib/grafana']
                }
            },
            'networks': {
                'app-network': {
                    'driver': 'bridge'
                }
            },
            'volumes': {
                'redis_data': {},
                'postgres_data': {},
                'prometheus_data': {},
                'grafana_data': {}
            }
        }

        return yaml.dump(compose, default_flow_style=False, sort_keys=False)

    @staticmethod
    def generate_helm_chart(app_name: str) -> Dict[str, str]:
        """Generate Helm chart."""

        files = {}

        # Chart.yaml
        files['Chart.yaml'] = f'''apiVersion: v2
name: {app_name}
description: A Helm chart for {app_name}
type: application
version: 1.0.0
appVersion: "1.0.0"
'''

        # values.yaml
        files['values.yaml'] = f'''replicaCount: 3

image:
  repository: {app_name}
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: LoadBalancer
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: {app_name}.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: {app_name}-tls
      hosts:
        - {app_name}.example.com

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

env:
  ENVIRONMENT: production
  LOG_LEVEL: INFO
'''

        return files


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate deployment configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Kubernetes manifests
  greenlang generate-deployment --platform kubernetes

  # Generate Terraform configuration
  greenlang generate-deployment --platform terraform

  # Generate Docker Compose for production
  greenlang generate-deployment --platform docker

  # Generate Helm chart
  greenlang generate-deployment --platform helm

  # Generate all
  greenlang generate-deployment --all
        """
    )

    parser.add_argument('--platform', choices=['kubernetes', 'terraform', 'docker', 'helm'],
                        help='Deployment platform')
    parser.add_argument('--all', action='store_true', help='Generate for all platforms')
    parser.add_argument('--app-name', default='greenlang-app', help='Application name')
    parser.add_argument('--output-dir', default='.', help='Output directory')

    args = parser.parse_args()

    if not args.platform and not args.all:
        parser.print_help()
        sys.exit(1)

    generator = DeploymentGenerator()
    output_dir = Path(args.output_dir)

    print("\nGenerating deployment configurations...\n")

    # Kubernetes
    if args.platform == 'kubernetes' or args.all:
        k8s_dir = output_dir / 'k8s'
        k8s_dir.mkdir(parents=True, exist_ok=True)

        manifests = generator.generate_kubernetes_manifests(args.app_name)
        for filename, content in manifests.items():
            file_path = k8s_dir / filename
            file_path.write_text(content, encoding='utf-8')
            print(f"Generated: {file_path}")

    # Terraform
    if args.platform == 'terraform' or args.all:
        tf_dir = output_dir / 'terraform'
        tf_dir.mkdir(parents=True, exist_ok=True)

        tf_files = generator.generate_terraform(args.app_name)
        for filename, content in tf_files.items():
            file_path = tf_dir / filename
            file_path.write_text(content, encoding='utf-8')
            print(f"Generated: {file_path}")

    # Docker Compose
    if args.platform == 'docker' or args.all:
        compose_content = generator.generate_docker_compose_production(args.app_name)
        file_path = output_dir / 'docker-compose.prod.yml'
        file_path.write_text(compose_content, encoding='utf-8')
        print(f"Generated: {file_path}")

    # Helm
    if args.platform == 'helm' or args.all:
        helm_dir = output_dir / 'helm' / args.app_name
        helm_dir.mkdir(parents=True, exist_ok=True)

        helm_files = generator.generate_helm_chart(args.app_name)
        for filename, content in helm_files.items():
            file_path = helm_dir / filename
            file_path.write_text(content, encoding='utf-8')
            print(f"Generated: {file_path}")

    print("\nDeployment configurations generated successfully!")
    print("\nNext steps:")
    print("  1. Review and customize the generated files")
    print("  2. Set up your cloud provider credentials")
    print("  3. Deploy using the appropriate tool")
    print()


if __name__ == '__main__':
    main()
