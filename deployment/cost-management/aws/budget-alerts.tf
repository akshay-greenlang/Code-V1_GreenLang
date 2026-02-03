# AWS Budget Alerts Configuration
# INFRA-001: Cost Management and Optimization
# Terraform configuration for AWS cost budgets and alerts

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "monthly_budget_limit" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 25000
}

variable "alert_email_addresses" {
  description = "Email addresses for budget alerts"
  type        = list(string)
  default     = ["finops@greenlang.io", "platform-team@greenlang.io"]
}

variable "slack_webhook_arn" {
  description = "ARN of SNS topic for Slack integration"
  type        = string
  default     = ""
}

variable "cost_center_budgets" {
  description = "Budgets per cost center"
  type = map(object({
    limit        = number
    alert_emails = list(string)
  }))
  default = {
    "CC-1001" = {
      limit        = 5000
      alert_emails = ["platform-team@greenlang.io"]
    }
    "CC-2001" = {
      limit        = 3000
      alert_emails = ["data-team@greenlang.io"]
    }
    "CC-3001" = {
      limit        = 8000
      alert_emails = ["ml-team@greenlang.io"]
    }
    "CC-4001" = {
      limit        = 2000
      alert_emails = ["sre-team@greenlang.io"]
    }
  }
}

# SNS Topic for budget alerts
resource "aws_sns_topic" "budget_alerts" {
  name = "greenlang-budget-alerts-${var.environment}"

  tags = {
    Environment = var.environment
    Project     = "GreenLang"
    ManagedBy   = "Terraform"
  }
}

# SNS Topic Policy
resource "aws_sns_topic_policy" "budget_alerts" {
  arn = aws_sns_topic.budget_alerts.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowBudgetsPublish"
        Effect = "Allow"
        Principal = {
          Service = "budgets.amazonaws.com"
        }
        Action   = "SNS:Publish"
        Resource = aws_sns_topic.budget_alerts.arn
      }
    ]
  })
}

# Email subscriptions
resource "aws_sns_topic_subscription" "budget_alert_emails" {
  for_each = toset(var.alert_email_addresses)

  topic_arn = aws_sns_topic.budget_alerts.arn
  protocol  = "email"
  endpoint  = each.value
}

# Overall Monthly Budget
resource "aws_budgets_budget" "monthly_total" {
  name         = "greenlang-monthly-total-${var.environment}"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_limit
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Project$GreenLang"]
  }

  # 50% threshold notification
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  # 80% threshold notification
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  # 100% threshold notification
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  # 80% forecasted threshold
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  # 100% forecasted threshold
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  tags = {
    Environment = var.environment
    Project     = "GreenLang"
    ManagedBy   = "Terraform"
  }
}

# EKS Cluster Budget
resource "aws_budgets_budget" "eks_cluster" {
  name         = "greenlang-eks-${var.environment}"
  budget_type  = "COST"
  limit_amount = 12000
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "Service"
    values = ["Amazon Elastic Kubernetes Service"]
  }

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Project$GreenLang"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  tags = {
    Environment = var.environment
    Project     = "GreenLang"
    Service     = "EKS"
    ManagedBy   = "Terraform"
  }
}

# RDS Database Budget
resource "aws_budgets_budget" "rds" {
  name         = "greenlang-rds-${var.environment}"
  budget_type  = "COST"
  limit_amount = 3000
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "Service"
    values = ["Amazon Relational Database Service"]
  }

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Project$GreenLang"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  tags = {
    Environment = var.environment
    Project     = "GreenLang"
    Service     = "RDS"
    ManagedBy   = "Terraform"
  }
}

# S3 Storage Budget
resource "aws_budgets_budget" "s3" {
  name         = "greenlang-s3-${var.environment}"
  budget_type  = "COST"
  limit_amount = 1000
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "Service"
    values = ["Amazon Simple Storage Service"]
  }

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Project$GreenLang"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  tags = {
    Environment = var.environment
    Project     = "GreenLang"
    Service     = "S3"
    ManagedBy   = "Terraform"
  }
}

# Data Transfer Budget
resource "aws_budgets_budget" "data_transfer" {
  name         = "greenlang-data-transfer-${var.environment}"
  budget_type  = "COST"
  limit_amount = 2000
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "UsageType"
    values = ["DataTransfer-Out-Bytes", "DataTransfer-Regional-Bytes"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 70
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  tags = {
    Environment = var.environment
    Project     = "GreenLang"
    Service     = "DataTransfer"
    ManagedBy   = "Terraform"
  }
}

# Cost Center Budgets
resource "aws_budgets_budget" "cost_centers" {
  for_each = var.cost_center_budgets

  name         = "greenlang-${each.key}-${var.environment}"
  budget_type  = "COST"
  limit_amount = each.value.limit
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:CostCenter$${each.key}"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 70
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = each.value.alert_emails
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 90
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = each.value.alert_emails
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = concat(each.value.alert_emails, var.alert_email_addresses)
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  tags = {
    Environment = var.environment
    Project     = "GreenLang"
    CostCenter  = each.key
    ManagedBy   = "Terraform"
  }
}

# Daily Spend Anomaly Budget
resource "aws_budgets_budget" "daily_anomaly" {
  name         = "greenlang-daily-anomaly-${var.environment}"
  budget_type  = "COST"
  limit_amount = 1000  # Daily threshold
  limit_unit   = "USD"
  time_unit    = "DAILY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Project$GreenLang"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 150  # 150% of daily budget = anomaly
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  tags = {
    Environment = var.environment
    Project     = "GreenLang"
    Type        = "AnomalyDetection"
    ManagedBy   = "Terraform"
  }
}

# Savings Plan Utilization Budget
resource "aws_budgets_budget" "savings_plan_utilization" {
  name              = "greenlang-sp-utilization-${var.environment}"
  budget_type       = "SAVINGS_PLANS_UTILIZATION"
  limit_amount      = 90  # Target 90% utilization
  limit_unit        = "PERCENTAGE"
  time_unit         = "MONTHLY"

  cost_types {
    include_credit             = false
    include_discount           = true
    include_other_subscription = true
    include_recurring          = true
    include_refund             = false
    include_subscription       = true
    include_support            = true
    include_tax                = true
    include_upfront            = true
    use_blended                = false
  }

  notification {
    comparison_operator        = "LESS_THAN"
    threshold                  = 80  # Alert if utilization drops below 80%
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  tags = {
    Environment = var.environment
    Project     = "GreenLang"
    Type        = "SavingsPlanUtilization"
    ManagedBy   = "Terraform"
  }
}

# Reserved Instance Utilization Budget
resource "aws_budgets_budget" "ri_utilization" {
  name              = "greenlang-ri-utilization-${var.environment}"
  budget_type       = "RI_UTILIZATION"
  limit_amount      = 90  # Target 90% utilization
  limit_unit        = "PERCENTAGE"
  time_unit         = "MONTHLY"

  cost_types {
    include_credit             = false
    include_discount           = true
    include_other_subscription = true
    include_recurring          = true
    include_refund             = false
    include_subscription       = true
    include_support            = true
    include_tax                = true
    include_upfront            = true
    use_blended                = false
  }

  notification {
    comparison_operator        = "LESS_THAN"
    threshold                  = 80  # Alert if utilization drops below 80%
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.alert_email_addresses
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }

  tags = {
    Environment = var.environment
    Project     = "GreenLang"
    Type        = "RIUtilization"
    ManagedBy   = "Terraform"
  }
}

# Outputs
output "sns_topic_arn" {
  description = "ARN of the budget alerts SNS topic"
  value       = aws_sns_topic.budget_alerts.arn
}

output "budget_ids" {
  description = "IDs of created budgets"
  value = {
    monthly_total      = aws_budgets_budget.monthly_total.id
    eks_cluster        = aws_budgets_budget.eks_cluster.id
    rds                = aws_budgets_budget.rds.id
    s3                 = aws_budgets_budget.s3.id
    data_transfer      = aws_budgets_budget.data_transfer.id
    daily_anomaly      = aws_budgets_budget.daily_anomaly.id
    sp_utilization     = aws_budgets_budget.savings_plan_utilization.id
    ri_utilization     = aws_budgets_budget.ri_utilization.id
    cost_centers       = { for k, v in aws_budgets_budget.cost_centers : k => v.id }
  }
}
