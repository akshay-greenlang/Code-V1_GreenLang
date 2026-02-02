output "log_group_names" { value = concat(aws_cloudwatch_log_group.eks[*].name, aws_cloudwatch_log_group.rds[*].name) }
output "alarm_arns" { value = [] }
output "sns_topic_arns" { value = { critical = try(aws_sns_topic.critical_alerts[0].arn, ""); warning = try(aws_sns_topic.warning_alerts[0].arn, "") } }
