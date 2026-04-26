package greenlang_guardrails

import rego.v1

test_safe_change_allowed if {
	allow with input as {"force_push": false, "delete_default_branch": false}
}

test_force_push_denied if {
	not allow with input as {"force_push": true}
}

test_default_branch_delete_denied if {
	not allow with input as {"delete_default_branch": true}
}
