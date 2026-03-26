package greenlang.v1.install

default allow = false

allow {
  input.pack.security.signed == true
  count(input.pack.security.signatures) > 0
}

deny_reason[msg] {
  not input.pack.security.signed
  msg := "v1 pack must be signed"
}

deny_reason[msg] {
  count(input.pack.security.signatures) == 0
  msg := "v1 pack must include at least one signature file"
}

