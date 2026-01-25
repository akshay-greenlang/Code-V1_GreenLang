output "vault_arn" { value = try(aws_backup_vault.main[0].arn, "") }
output "plan_id" { value = try(aws_backup_plan.main[0].id, "") }
output "vault_name" { value = try(aws_backup_vault.main[0].name, "") }
