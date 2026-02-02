output "provenance_bucket_id" { value = aws_s3_bucket.main["provenance"].id }
output "provenance_bucket_arn" { value = aws_s3_bucket.main["provenance"].arn }
output "raw_data_bucket_id" { value = aws_s3_bucket.main["raw_data"].id }
output "raw_data_bucket_arn" { value = aws_s3_bucket.main["raw_data"].arn }
output "reports_bucket_id" { value = aws_s3_bucket.main["reports"].id }
output "reports_bucket_arn" { value = aws_s3_bucket.main["reports"].arn }
output "all_bucket_arns" { value = [for b in aws_s3_bucket.main : b.arn] }
