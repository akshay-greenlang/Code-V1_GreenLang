# =============================================================================
# GreenLang Grafana Module - Outputs
# GreenLang Climate OS | OBS-002
# =============================================================================

# ---------------------------------------------------------------------------
# RDS Outputs
# ---------------------------------------------------------------------------

output "db_endpoint" {
  description = "RDS instance endpoint (hostname)"
  value       = aws_db_instance.grafana.address
}

output "db_port" {
  description = "RDS instance port"
  value       = aws_db_instance.grafana.port
}

output "db_name" {
  description = "Grafana database name"
  value       = aws_db_instance.grafana.db_name
}

output "db_username" {
  description = "Grafana database username"
  value       = aws_db_instance.grafana.username
}

output "db_secret_arn" {
  description = "ARN of the Secrets Manager secret containing DB credentials"
  value       = aws_secretsmanager_secret.grafana_db.arn
}

output "db_instance_id" {
  description = "RDS instance identifier"
  value       = aws_db_instance.grafana.id
}

# ---------------------------------------------------------------------------
# S3 Outputs
# ---------------------------------------------------------------------------

output "s3_bucket_name" {
  description = "Name of the S3 bucket for Grafana image storage"
  value       = aws_s3_bucket.grafana_images.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for Grafana image storage"
  value       = aws_s3_bucket.grafana_images.arn
}

# ---------------------------------------------------------------------------
# IAM Outputs
# ---------------------------------------------------------------------------

output "iam_role_arn" {
  description = "ARN of the Grafana IRSA IAM role"
  value       = aws_iam_role.grafana.arn
}

output "iam_role_name" {
  description = "Name of the Grafana IRSA IAM role"
  value       = aws_iam_role.grafana.name
}

# ---------------------------------------------------------------------------
# Security Group Outputs
# ---------------------------------------------------------------------------

output "security_group_id" {
  description = "Security group ID for the Grafana RDS instance"
  value       = aws_security_group.grafana_db.id
}

# ---------------------------------------------------------------------------
# Kubernetes Outputs
# ---------------------------------------------------------------------------

output "k8s_secret_name" {
  description = "Name of the Kubernetes secret containing DB credentials"
  value       = kubernetes_secret.grafana_db.metadata[0].name
}
