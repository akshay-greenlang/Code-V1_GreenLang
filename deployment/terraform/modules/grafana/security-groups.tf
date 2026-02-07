# =============================================================================
# GreenLang Grafana Module - Security Groups
# GreenLang Climate OS | OBS-002
# =============================================================================

# ---------------------------------------------------------------------------
# Grafana DB Security Group
# ---------------------------------------------------------------------------

resource "aws_security_group" "grafana_db" {
  name_prefix = "${local.name_prefix}-db-"
  description = "Security group for Grafana RDS PostgreSQL instance"
  vpc_id      = var.vpc_id

  tags = merge(local.default_tags, {
    Name = "${local.name_prefix}-db-sg"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Allow PostgreSQL access from EKS worker nodes
resource "aws_security_group_rule" "grafana_db_ingress_eks" {
  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  description              = "Allow PostgreSQL access from EKS nodes"
  security_group_id        = aws_security_group.grafana_db.id
  source_security_group_id = var.eks_node_security_group_id
}
