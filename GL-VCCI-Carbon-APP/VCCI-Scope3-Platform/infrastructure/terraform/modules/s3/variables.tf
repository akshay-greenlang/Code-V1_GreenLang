variable "project_name" { type = string }
variable "environment" { type = string }
variable "enable_versioning" { type = bool }
variable "enable_replication" { type = bool }
variable "replication_region" { type = string }
variable "lifecycle_glacier_transition_days" { type = number }
variable "lifecycle_ia_transition_days" { type = number }
variable "lifecycle_expiration_days" { type = number }
variable "kms_key_arn" { type = string }
variable "tags" { type = map(string); default = {} }
