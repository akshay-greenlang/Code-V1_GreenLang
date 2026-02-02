output "s3_access_role_arn" { value = aws_iam_role.s3_access.arn }
output "rds_access_role_arn" { value = aws_iam_role.rds_access.arn }
output "cloudwatch_logs_policy_arn" { value = aws_iam_policy.cloudwatch_logs.arn }
output "eks_cluster_role_arn" { value = aws_iam_role.s3_access.arn }
output "eks_node_role_arn" { value = aws_iam_role.s3_access.arn }
