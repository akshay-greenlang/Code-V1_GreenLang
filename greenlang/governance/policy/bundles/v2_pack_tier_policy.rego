package greenlang.v2.pack_tier_policy

default allow = true

deny[msg] {
  input.contract_version == "2.0"
  tier := input.metadata.quality_tier
  tier == "supported" or tier == "regulated-critical"
  not input.security.signed
  msg := sprintf("tier %v requires signed pack", [tier])
}

deny[msg] {
  input.contract_version == "2.0"
  input.security.signed
  count(input.security.signatures) == 0
  msg := "signed pack must include at least one signature artifact"
}

deny[msg] {
  input.contract_version == "2.0"
  tier := input.metadata.quality_tier
  tier == "candidate" or tier == "supported" or tier == "regulated-critical"
  not input.metadata.owner_team
  msg := sprintf("tier %v requires owner_team", [tier])
}

deny[msg] {
  input.contract_version == "2.0"
  tier := input.metadata.quality_tier
  tier == "candidate" or tier == "supported" or tier == "regulated-critical"
  not input.metadata.support_channel
  msg := sprintf("tier %v requires support_channel", [tier])
}

deny[msg] {
  input.contract_version == "2.0"
  tier := input.metadata.quality_tier
  tier == "candidate" or tier == "supported" or tier == "regulated-critical"
  not input.evidence.docs_contract
  msg := sprintf("tier %v requires docs_contract evidence", [tier])
}

deny[msg] {
  input.contract_version == "2.0"
  tier := input.metadata.quality_tier
  tier == "supported" or tier == "regulated-critical"
  not input.evidence.security_scan
  msg := sprintf("tier %v requires security_scan evidence", [tier])
}

deny[msg] {
  input.contract_version == "2.0"
  input.metadata.quality_tier == "regulated-critical"
  not input.evidence.determinism_report
  msg := "tier regulated-critical requires determinism_report evidence"
}

deny[msg] {
  input.contract_version == "2.0"
  tier := input.metadata.quality_tier
  expected := {
    "experimental": "pilot-approved",
    "candidate": "candidate-approved",
    "supported": "supported-approved",
    "regulated-critical": "regulated-approved",
  }[tier]
  expected != ""
  input.promotion_status != expected
  msg := sprintf("tier %v requires promotion_status=%v", [tier, expected])
}

