variable "project_name" { type = string }
variable "environment" { type = string }
variable "cluster_name" { type = string }
variable "oidc_issuer_url" { type = string }
variable "s3_bucket_arns" { type = list(string) }
variable "tags" { type = map(string); default = {} }
