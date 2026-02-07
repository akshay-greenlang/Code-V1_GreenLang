# =============================================================================
# GreenLang Prometheus Stack Module - Provider Versions
# GreenLang Climate OS | OBS-001
# =============================================================================
# Defines required Terraform and provider versions for the Prometheus stack
# deployment. This module provisions Prometheus HA with Thanos for long-term
# storage, Alertmanager, and PushGateway.
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.12"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.25"
    }
  }
}
