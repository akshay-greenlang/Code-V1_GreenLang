package greenlang.v2.egress_controls

default allow = true

deny[msg] {
  input.contract_version == "2.0"
  input.workflow_tier == "regulated-critical"
  not input.egress.allowlist
  msg := "regulated-critical workflows require explicit egress allowlist"
}

deny[msg] {
  input.contract_version == "2.0"
  input.workflow_tier == "regulated-critical"
  destination := input.egress.destination
  not destination_allowed(destination, input.egress.allowlist)
  msg := sprintf("egress destination '%v' is not allowed", [destination])
}

destination_allowed(dest, allowlist) {
  allowlist[_] == dest
}
