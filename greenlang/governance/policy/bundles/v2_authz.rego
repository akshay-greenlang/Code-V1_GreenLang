package greenlang.v2.authz

default allow = false

allow {
  input.contract_version == "2.0"
  input.authz.approved == true
  input.authz.role != ""
}

deny[msg] {
  input.contract_version == "2.0"
  not input.authz.approved
  msg := "authz approval is required"
}
