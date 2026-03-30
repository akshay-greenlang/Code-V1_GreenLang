package greenlang.v2.data_controls

default allow = false

allow {
  input.contract_version == "2.0"
  input.data_classification != ""
  input.retention_days > 0
  input.data_residency != ""
}

deny[msg] {
  input.contract_version == "2.0"
  input.data_classification == ""
  msg := "data classification is required"
}

deny[msg] {
  input.contract_version == "2.0"
  input.retention_days <= 0
  msg := "retention_days must be positive"
}

deny[msg] {
  input.contract_version == "2.0"
  input.data_residency == ""
  msg := "data residency must be declared"
}
