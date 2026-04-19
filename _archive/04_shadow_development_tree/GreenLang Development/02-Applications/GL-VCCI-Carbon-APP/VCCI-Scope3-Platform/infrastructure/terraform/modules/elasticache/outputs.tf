output "cluster_id" { value = aws_elasticache_replication_group.main.id }
output "cluster_arn" { value = aws_elasticache_replication_group.main.arn }
output "cluster_configuration_endpoint" { value = aws_elasticache_replication_group.main.configuration_endpoint_address }
output "cluster_reader_endpoint" { value = aws_elasticache_replication_group.main.reader_endpoint_address }
output "cluster_port" { value = 6379 }
output "security_group_id" { value = aws_security_group.redis.id }
output "primary_endpoint_address" { value = aws_elasticache_replication_group.main.primary_endpoint_address }
