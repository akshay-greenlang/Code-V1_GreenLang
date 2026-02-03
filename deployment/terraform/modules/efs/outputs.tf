#------------------------------------------------------------------------------
# AWS EFS Module - Outputs
# GreenLang Infrastructure
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# EFS File System Outputs
#------------------------------------------------------------------------------

output "file_system_id" {
  description = "ID of the EFS file system"
  value       = aws_efs_file_system.main.id
}

output "file_system_arn" {
  description = "ARN of the EFS file system"
  value       = aws_efs_file_system.main.arn
}

output "file_system_dns_name" {
  description = "DNS name of the EFS file system"
  value       = aws_efs_file_system.main.dns_name
}

output "file_system_size_in_bytes" {
  description = "Latest known metered size of the file system"
  value       = aws_efs_file_system.main.size_in_bytes
}

output "file_system_performance_mode" {
  description = "Performance mode of the file system"
  value       = aws_efs_file_system.main.performance_mode
}

output "file_system_throughput_mode" {
  description = "Throughput mode of the file system"
  value       = aws_efs_file_system.main.throughput_mode
}

#------------------------------------------------------------------------------
# Mount Target Outputs
#------------------------------------------------------------------------------

output "mount_target_ids" {
  description = "Map of subnet IDs to mount target IDs"
  value       = { for k, v in aws_efs_mount_target.main : k => v.id }
}

output "mount_target_ips" {
  description = "Map of subnet IDs to mount target IP addresses"
  value       = { for k, v in aws_efs_mount_target.main : k => v.ip_address }
}

output "mount_target_dns_names" {
  description = "Map of subnet IDs to mount target DNS names"
  value       = { for k, v in aws_efs_mount_target.main : k => v.dns_name }
}

output "mount_target_network_interface_ids" {
  description = "Map of subnet IDs to mount target network interface IDs"
  value       = { for k, v in aws_efs_mount_target.main : k => v.network_interface_id }
}

#------------------------------------------------------------------------------
# Access Point Outputs
#------------------------------------------------------------------------------

output "access_point_artifacts_id" {
  description = "ID of the artifacts access point"
  value       = aws_efs_access_point.artifacts.id
}

output "access_point_artifacts_arn" {
  description = "ARN of the artifacts access point"
  value       = aws_efs_access_point.artifacts.arn
}

output "access_point_models_id" {
  description = "ID of the models access point"
  value       = aws_efs_access_point.models.id
}

output "access_point_models_arn" {
  description = "ARN of the models access point"
  value       = aws_efs_access_point.models.arn
}

output "access_point_shared_id" {
  description = "ID of the shared access point"
  value       = aws_efs_access_point.shared.id
}

output "access_point_shared_arn" {
  description = "ARN of the shared access point"
  value       = aws_efs_access_point.shared.arn
}

output "access_point_tmp_id" {
  description = "ID of the tmp access point"
  value       = aws_efs_access_point.tmp.id
}

output "access_point_tmp_arn" {
  description = "ARN of the tmp access point"
  value       = aws_efs_access_point.tmp.arn
}

output "access_points" {
  description = "Map of all access point IDs and ARNs"
  value = {
    artifacts = {
      id  = aws_efs_access_point.artifacts.id
      arn = aws_efs_access_point.artifacts.arn
    }
    models = {
      id  = aws_efs_access_point.models.id
      arn = aws_efs_access_point.models.arn
    }
    shared = {
      id  = aws_efs_access_point.shared.id
      arn = aws_efs_access_point.shared.arn
    }
    tmp = {
      id  = aws_efs_access_point.tmp.id
      arn = aws_efs_access_point.tmp.arn
    }
  }
}

output "additional_access_point_ids" {
  description = "Map of additional access point names to IDs"
  value       = { for k, v in aws_efs_access_point.additional : k => v.id }
}

output "additional_access_point_arns" {
  description = "Map of additional access point names to ARNs"
  value       = { for k, v in aws_efs_access_point.additional : k => v.arn }
}

#------------------------------------------------------------------------------
# Security Group Outputs
#------------------------------------------------------------------------------

output "security_group_id" {
  description = "ID of the EFS security group"
  value       = aws_security_group.efs.id
}

output "security_group_arn" {
  description = "ARN of the EFS security group"
  value       = aws_security_group.efs.arn
}

output "security_group_name" {
  description = "Name of the EFS security group"
  value       = aws_security_group.efs.name
}

#------------------------------------------------------------------------------
# KMS Key Outputs
#------------------------------------------------------------------------------

output "kms_key_id" {
  description = "ID of the KMS key used for EFS encryption"
  value       = var.create_kms_key ? aws_kms_key.efs[0].key_id : var.kms_key_id
}

output "kms_key_arn" {
  description = "ARN of the KMS key used for EFS encryption"
  value       = var.create_kms_key ? aws_kms_key.efs[0].arn : var.kms_key_id
}

output "kms_key_alias" {
  description = "Alias of the KMS key"
  value       = var.create_kms_key ? aws_kms_alias.efs[0].name : null
}

#------------------------------------------------------------------------------
# Backup Outputs
#------------------------------------------------------------------------------

output "backup_vault_id" {
  description = "ID of the backup vault"
  value       = var.enable_backup && var.create_backup_vault ? aws_backup_vault.efs[0].id : null
}

output "backup_vault_arn" {
  description = "ARN of the backup vault"
  value       = var.enable_backup && var.create_backup_vault ? aws_backup_vault.efs[0].arn : null
}

output "backup_vault_name" {
  description = "Name of the backup vault"
  value       = var.enable_backup && var.create_backup_vault ? aws_backup_vault.efs[0].name : var.backup_vault_name
}

output "backup_plan_id" {
  description = "ID of the backup plan"
  value       = var.enable_backup && var.create_backup_plan ? aws_backup_plan.efs[0].id : null
}

output "backup_plan_arn" {
  description = "ARN of the backup plan"
  value       = var.enable_backup && var.create_backup_plan ? aws_backup_plan.efs[0].arn : null
}

output "backup_selection_id" {
  description = "ID of the backup selection"
  value       = var.enable_backup && var.create_backup_plan ? aws_backup_selection.efs[0].id : null
}

#------------------------------------------------------------------------------
# IAM Outputs
#------------------------------------------------------------------------------

output "efs_access_policy_arn" {
  description = "ARN of the EFS access IAM policy"
  value       = aws_iam_policy.efs_access.arn
}

output "efs_access_policy_name" {
  description = "Name of the EFS access IAM policy"
  value       = aws_iam_policy.efs_access.name
}

output "irsa_role_arn" {
  description = "ARN of the IRSA role for EKS pods"
  value       = var.eks_cluster_name != null ? aws_iam_role.irsa[0].arn : null
}

output "irsa_role_name" {
  description = "Name of the IRSA role for EKS pods"
  value       = var.eks_cluster_name != null ? aws_iam_role.irsa[0].name : null
}

output "backup_role_arn" {
  description = "ARN of the AWS Backup IAM role"
  value       = var.enable_backup && var.create_backup_plan ? aws_iam_role.backup[0].arn : null
}

output "backup_role_name" {
  description = "Name of the AWS Backup IAM role"
  value       = var.enable_backup && var.create_backup_plan ? aws_iam_role.backup[0].name : null
}

#------------------------------------------------------------------------------
# Replication Outputs
#------------------------------------------------------------------------------

output "replication_configuration_id" {
  description = "ID of the EFS replication configuration"
  value       = var.enable_replication ? aws_efs_replication_configuration.main[0].original_source_file_system_arn : null
}

output "replication_destination_file_system_id" {
  description = "File system ID of the replication destination"
  value       = var.enable_replication ? aws_efs_replication_configuration.main[0].destination[0].file_system_id : null
}

#------------------------------------------------------------------------------
# CloudWatch Alarm Outputs
#------------------------------------------------------------------------------

output "cloudwatch_alarm_burst_credit_arn" {
  description = "ARN of the burst credit balance CloudWatch alarm"
  value       = var.enable_cloudwatch_alarms ? aws_cloudwatch_metric_alarm.burst_credit_balance[0].arn : null
}

output "cloudwatch_alarm_percent_io_arn" {
  description = "ARN of the percent IO limit CloudWatch alarm"
  value       = var.enable_cloudwatch_alarms ? aws_cloudwatch_metric_alarm.percent_io_limit[0].arn : null
}

#------------------------------------------------------------------------------
# Kubernetes Integration Outputs
#------------------------------------------------------------------------------

output "storage_class_name" {
  description = "Name of the Kubernetes StorageClass"
  value       = var.eks_cluster_name != null ? "efs-sc" : null
}

output "efs_mount_command" {
  description = "Command to mount EFS on EC2 instances"
  value       = "sudo mount -t efs -o tls ${aws_efs_file_system.main.id}:/ /mnt/efs"
}

output "efs_mount_command_with_access_point" {
  description = "Command to mount EFS with access point on EC2 instances"
  value       = "sudo mount -t efs -o tls,accesspoint=${aws_efs_access_point.shared.id} ${aws_efs_file_system.main.id}:/ /mnt/efs"
}
