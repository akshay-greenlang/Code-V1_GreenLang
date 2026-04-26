package greenlang_guardrails

import rego.v1

default allow := false

allow if {
	not dangerous_change
}

dangerous_change if {
	input.force_push == true
}

dangerous_change if {
	input.delete_default_branch == true
}
