package greenlang.v2.pack_trust

default allow = true

deny[msg] {
  input.contract_version == "2.0"
  tier := input.metadata.quality_tier
  tier == "supported" or tier == "regulated-critical"
  not input.security.signed
  msg := sprintf("tier %v requires trusted signed pack", [tier])
}

deny[msg] {
  input.contract_version == "2.0"
  input.security.signed
  count(input.security.signatures) == 0
  msg := "trusted pack must include at least one signature artifact"
}
